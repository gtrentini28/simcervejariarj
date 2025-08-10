# sprayball_sim.py
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from math import pi, cos, sin

st.set_page_config(layout="wide", page_title="Sprayball Tank Cleaner Simulator")

st.title("Simulador Interativo — Sprayball / Cobertura em Tanque Cilíndrico-Cônico Vertical")
st.markdown("Modelo físico 3D simplificado: gera mapa de cobertura (m³/s·m⁻²) sobre superfície interna do tanque.")

# -----------------------
# Helpers geométricos
# -----------------------
def unit(v):
    n = np.linalg.norm(v)
    return v / n if n != 0 else v

def ray_cylinder_intersection(ray_o, ray_d, cyl_radius, cyl_z_min, cyl_z_max):
    # ray: p = o + t d
    # cylinder: x^2 + y^2 = r^2, z in [zmin, zmax]
    ox, oy, oz = ray_o
    dx, dy, dz = ray_d
    a = dx*dx + dy*dy
    b = 2*(ox*dx + oy*dy)
    c = ox*ox + oy*oy - cyl_radius*cyl_radius
    ts = []
    if abs(a) > 1e-12:
        disc = b*b - 4*a*c
        if disc >= 0:
            t1 = (-b - np.sqrt(disc)) / (2*a)
            t2 = (-b + np.sqrt(disc)) / (2*a)
            for t in (t1, t2):
                if t > 1e-9:
                    z = oz + t*dz
                    if cyl_z_min - 1e-9 <= z <= cyl_z_max + 1e-9:
                        ts.append(t)
    return min(ts) if ts else None

def ray_cone_intersection(ray_o, ray_d, cone_apex, cone_axis, cone_angle, h_cone):
    # Slender conical surface (apex at cone_apex, axis points downwards into cone_axis unit vector,
    # half-angle = cone_angle, truncated up to height h_cone along axis)
    # Solve parametric--we'll transform coords so cone axis = z, apex at origin
    # Build orthonormal basis with cone_axis as +z
    ez = unit(cone_axis)
    # choose ex arbitrary perpendicular
    tmp = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(tmp, ez)) > 0.9:
        tmp = np.array([0.0, 1.0, 0.0])
    ex = unit(np.cross(tmp, ez))
    ey = np.cross(ez, ex)
    # transform
    def to_local(p):
        v = p - cone_apex
        return np.array([np.dot(v, ex), np.dot(v, ey), np.dot(v, ez)])
    o_loc = to_local(ray_o)
    d_loc = np.array([np.dot(ray_d, ex), np.dot(ray_d, ey), np.dot(ray_d, ez)])
    # cone equation: x^2 + y^2 = (z * tan(alpha))^2, for z in [0, h_cone]
    tan_a = np.tan(cone_angle)
    A = d_loc[0]**2 + d_loc[1]**2 - (tan_a**2) * (d_loc[2]**2)
    B = 2*(o_loc[0]*d_loc[0] + o_loc[1]*d_loc[1] - (tan_a**2)*o_loc[2]*d_loc[2])
    C = o_loc[0]**2 + o_loc[1]**2 - (tan_a**2)*(o_loc[2]**2)
    if abs(A) < 1e-12:
        return None
    disc = B*B - 4*A*C
    if disc < 0:
        return None
    t1 = (-B - np.sqrt(disc)) / (2*A)
    t2 = (-B + np.sqrt(disc)) / (2*A)
    candidates = []
    for t in (t1, t2):
        if t > 1e-9:
            p = o_loc + t*d_loc
            z = p[2]
            if 0 <= z <= h_cone:
                candidates.append(t)
    return min(candidates) if candidates else None

# -----------------------
# UI inputs
# -----------------------
st.sidebar.header("Tanque (cilíndrico + cone)")
tank_d = st.sidebar.number_input("Diâmetro interno (m)", value=2.0, min_value=0.1, step=0.1)
tank_h_cyl = st.sidebar.number_input("Altura do corpo cilíndrico (m)", value=3.0, min_value=0.1, step=0.1)
cone_height = st.sidebar.number_input("Altura do cone inferior (m)", value=0.6, min_value=0.0, step=0.05)

st.sidebar.header("Sprayball / Bocal")
nozzle_height = st.sidebar.number_input("Altura do sprayball acima do fundo (m)", value=2.5, min_value=0.0, step=0.01)
nozzle_offset_radial = st.sidebar.number_input("Offset radial do bocal (m) — 0 = centro", value=0.0, min_value=0.0, max_value=tank_d/2-0.001, step=0.01)
cone_angle_deg = st.sidebar.slider("Ângulo total do cone (graus)", min_value=20.0, max_value=160.0, value=110.0)
flow_lpm = st.sidebar.number_input("Vazão (L/min)", value=200.0, min_value=0.1, step=1.0)
pressure_bar = st.sidebar.number_input("Pressão de alimentação (bar)", value=1.0, min_value=0.0, step=0.1)
rho = st.sidebar.number_input("Densidade do fluido (kg/m³)", value=1000.0)

st.sidebar.header("Modelagem e resolução")
n_az = st.sidebar.slider("Resolução azimutal (amostras)", 36, 720, 180)
n_pol = st.sidebar.slider("Resolução polar (radial do cone)", 6, 120, 40)
rotate = st.sidebar.checkbox("Modelo rotativo (média temporal)", value=True)
rpm = st.sidebar.slider("Velocidade de rotação (RPM) — só se rotativo", 0.0, 120.0, 30.0) if rotate else 0.0

st.sidebar.markdown("---")
st.sidebar.write("Obs: Q e pressão ambos editáveis; relação Q↔P não é imposta (uso direto do Q informado).")

# Derived
Q_m3s = flow_lpm / 1000.0 / 60.0
half_angle = np.deg2rad(cone_angle_deg/2.0)
Omega = 2.0 * pi * (1.0 - cos(half_angle))  # solid angle of cone
nozzle_z = cone_height + nozzle_height  # global coordinate z=0 at tank bottom apex? We'll define bottom apex at z=0

# Tank geometry coordinates:
# bottom apex is at z=0, cone extends upward to z=cone_height, then cylinder up to cone_height + tank_h_cyl
z_min = 0.0
z_cone_top = cone_height
z_cyl_top = cone_height + tank_h_cyl
radius = tank_d / 2.0

# nozzle position: centered at x=y offset via radial offset at azimuth 0
nozzle_x = nozzle_offset_radial
nozzle_y = 0.0

# Prepare surface mesh of tank internal surface (parametric)
# Cylinder mesh (theta, z)
res_theta = 180
res_z = 120
theta_vals = np.linspace(0, 2*pi, res_theta)
z_vals_cyl = np.linspace(z_cone_top, z_cyl_top, res_z)
TH, ZZ = np.meshgrid(theta_vals, z_vals_cyl)
Xc = radius * np.cos(TH)
Yc = radius * np.sin(TH)
Zc = ZZ
# normals on cylinder = (cos, sin, 0)
Nc = np.stack([np.cos(TH), np.sin(TH), np.zeros_like(TH)], axis=-1)

# Cone mesh (ranging z from 0..cone_height)
res_theta_cone = 180
res_z_cone = 60
theta_c = np.linspace(0, 2*pi, res_theta_cone)
z_c = np.linspace(0.0, cone_height, res_z_cone)
TC, ZC = np.meshgrid(theta_c, z_c)
# radius at z: linear from 0 at apex to R at z=cone_height
r_at_z = (ZC / cone_height) * radius
Xcon = r_at_z * np.cos(TC)
Ycon = r_at_z * np.sin(TC)
Zcon = ZC
# normals for cone: compute via geometry
# cone generatrix angle: phi, its normal vector in local simple form:
# For point (x,y,z) normal ~ (x, y, -r'/?), we'll compute numerically
Nc_cone = np.zeros((Zcon.shape[0], Zcon.shape[1], 3))
for i in range(Zcon.shape[0]):
    for j in range(Zcon.shape[1]):
        # small finite diff to approximate normal: cross product of tangents
        p = np.array([Xcon[i,j], Ycon[i,j], Zcon[i,j]])
        # param derivatives
        dt = 1e-3
        dz = 1e-3
        p_theta = np.array([ (r_at_z[i,j])*(-np.sin(TC[i,j]))*1.0 , (r_at_z[i,j])*(np.cos(TC[i,j]))*1.0 , 0.0])
        # derivative wrt z:
        if cone_height > 0:
            dr_dz = radius / cone_height
        else:
            dr_dz = 0.0
        p_z = np.array([ dr_dz * np.cos(TC[i,j]) , dr_dz * np.sin(TC[i,j]) , 1.0 ])
        nvec = np.cross(p_theta, p_z)
        Nc_cone[i,j,:] = unit(nvec)

# Prepare accumulation grids
surf_points = []
surf_normals = []
surf_u = []  # parameter for plotting mapping
surf_v = []

# Cylinder points flatten
for i in range(Xc.shape[0]):
    for j in range(Xc.shape[1]):
        surf_points.append(np.array([Xc[i,j], Yc[i,j], Zc[i,j]]))
        surf_normals.append(Nc[i,j,:])
        surf_u.append(Xc[i,j])
        surf_v.append(Zc[i,j])

# Cone points flatten
for i in range(Xcon.shape[0]):
    for j in range(Xcon.shape[1]):
        surf_points.append(np.array([Xcon[i,j], Ycon[i,j], Zcon[i,j]]))
        surf_normals.append(Nc_cone[i,j,:])
        surf_u.append(Xcon[i,j])
        surf_v.append(Zcon[i,j])

Nsurf = len(surf_points)
flux_accum = np.zeros(Nsurf)  # m^3/s per m^2 (flux density) accumulated

# Ray sampling in cone: polar theta (azimuth) and polar angle phi (0..half_angle)
azimuths = np.linspace(0, 2*pi, n_az, endpoint=False)
polars = np.linspace(0, half_angle, n_pol)

# For rotative: average over rotated azimuth offsets
n_rot_samples = int(max(1, min(60, round(rpm/5)))) if rotate else 1
rot_offsets = np.linspace(0, 2*pi, n_rot_samples, endpoint=False) if rotate and rpm>0 else [0.0]

# Precompute single-ray contribution weight: each sample represents dOmega ~ (sin(phi) dphi dpsi)
# We'll use quadrature weights
phi_vals = (polars[:-1] + polars[1:]) / 2.0 if len(polars)>1 else polars
# Simpler: use direct grid midpoints and weights
if len(polars) > 1:
    phi_centers = (polars[:-1] + polars[1:]) / 2.0
    phi_widths = (polars[1:] - polars[:-1])
else:
    phi_centers = polars
    phi_widths = [polars[0] if len(polars)==1 else half_angle]

# Build list of sample directions
dir_list = []
domega_list = []
for rot in rot_offsets:
    for a in azimuths:
        for idx, phi in enumerate(phi_centers):
            # sample direction in nozzle local coords: take azimuth a + rot offset
            psi = a + rot
            sx = np.sin(phi) * np.cos(psi)
            sy = np.sin(phi) * np.sin(psi)
            sz = np.cos(phi)
            dir_list.append(unit(np.array([sx, sy, -sz])))  # minus z so cone points downward (toward tank)
            # weight: dOmega = sin(phi) dphi dpsi ; here dphi = phi_widths[idx], dpsi = 2pi/n_az
            dphi = phi_widths[idx]
            dpsi = 2*pi / n_az
            domega = np.sin(phi) * dphi * dpsi
            domega_list.append(domega / len(rot_offsets))  # average if rotated multiple samples

# Cast rays from nozzle
nozzle_pos = np.array([nozzle_x, nozzle_y, nozzle_z])
for k, (dvec, domega) in enumerate(zip(dir_list, domega_list)):
    # scale domega so total sums to Omega (sanity)
    # ray intersects tank internal surfaces: check cylinder then cone
    t_cyl = ray_cylinder_intersection(nozzle_pos, dvec, radius, z_cone_top, z_cyl_top)
    # cone apex at (0,0,0), axis pointing +z (from apex upward)
    t_cone = ray_cone_intersection(nozzle_pos, dvec, np.array([0.0,0.0,0.0]), np.array([0.0,0.0,1.0]), half_angle, cone_height)
    t_candidates = []
    if t_cyl is not None:
        t_candidates.append(('cyl', t_cyl))
    if t_cone is not None:
        t_candidates.append(('cone', t_cone))
    if not t_candidates:
        continue
    typ, tmin = min(t_candidates, key=lambda x: x[1])
    hit_p = nozzle_pos + tmin * dvec
    # find nearest surf index
    # brute force nearest point in surf_points
    dists = np.linalg.norm(np.array(surf_points) - hit_p.reshape(1,3), axis=1)
    idx = int(np.argmin(dists))
    r = np.linalg.norm(hit_p - nozzle_pos)
    nvec = surf_normals[idx]
    cos_inc = abs(np.dot(nvec, -dvec))  # angle between ray (into surface) and normal
    if cos_inc < 1e-6:
        continue
    # local flux density contribution: q_omega / (r^2 * cos_inc) * domega
    q_omega = Q_m3s / Omega if Omega > 0 else 0.0
    # volume_flow hitting this patch = q_omega * domega
    dQ = q_omega * domega
    # patch area represented by this sample ~ r^2 * cos_inc * ??? but we will convert to flux density:
    # local flux density [m3/s per m2] contribution:
    flux_density = 0.0
    if r > 0:
        flux_density = dQ / (r*r * cos_inc)
    flux_accum[idx] += flux_density

# Now we have flux density per mesh point. Convert to arrays for plotting
flux_arr = flux_accum  # m3/s per m2
# Convert to L/min per m2 for user-friendly units
flux_Lpm_m2 = flux_arr * 1000.0 * 60.0

# Prepare plotting arrays: reuse surf_points order
X = np.array([p[0] for p in surf_points])
Y = np.array([p[1] for p in surf_points])
Z = np.array([p[2] for p in surf_points])
C = flux_Lpm_m2

# Basic metrics
covered_cells = np.count_nonzero(C > 1e-6)
coverage_pct = 100.0 * covered_cells / len(C)
avg_flux = np.mean(C)
max_flux = np.max(C)

# Display numeric results
col1, col2, col3 = st.columns(3)
col1.metric("Cobertura (%)", f"{coverage_pct:.1f}%")
col2.metric("Fluxo médio (L/min·m²)", f"{avg_flux:.3f}")
col3.metric("Fluxo máximo (L/min·m²)", f"{max_flux:.3f}")

st.markdown("### Visualização 3D — mapa de fluxo (L/min·m²) na superfície interna")
# Plot with scatter3d colored
fig = go.Figure(data=[
    go.Mesh3d(
        x=X, y=Y, z=Z,
        intensity=C,
        colorscale='Viridis',
        showscale=True,
        opacity=0.9,
        alphahull=0
    )
])
# add nozzle marker
fig.add_trace(go.Scatter3d(x=[nozzle_pos[0]], y=[nozzle_pos[1]], z=[nozzle_pos[2]],
                           mode='markers', marker=dict(size=5, color='red'), name='Nozzle'))

fig.update_layout(scene=dict(aspectmode='data',
                             xaxis_title="x (m)", yaxis_title="y (m)", zaxis_title="z (m)"),
                  margin=dict(l=0,r=0,t=30,b=0),
                  height=700)
st.plotly_chart(fig, use_container_width=True)

st.markdown("### Observações / interpretação")
st.write("""
- A escala mostra fluxo volumétrico específico recebido pela superfície (L/min por m²). Zonas em azul claro/zero são pontos cegos.
- Use rotação (rotativo) para reduzir pontos cegos; velocidade rpm afeta apenas o número de amostragens temporais (não afeta fisicamente a geometria do cone).
- Para dimensionamento real, compare energia de impacto/velocidade de jato e tempo de exposição para remover sujidade.
- Modelo simplificado: assume distribuição uniforme por ângulo sólido e não resolve gotas individuais.
""")