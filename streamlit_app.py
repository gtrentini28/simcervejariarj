# brew_piping_simulator.py
import streamlit as st
import numpy as np
from math import pi, log10, sqrt
import matplotlib.pyplot as plt

g = 9.80665

def haaland_f(Re, D, eps):
    # Haaland explicit approximation for friction factor f
    if Re <= 0:
        return np.nan
    term = (eps / D) / 3.7
    inner = term**1.11 + 6.9 / Re
    inv_sqrt_f = -1.8 * np.log10(inner)
    f = 1.0 / (inv_sqrt_f**2)
    return f

def friction_factor(Re, D, eps):
    if Re < 2300:
        return 64.0 / Re
    else:
        return haaland_f(Re, D, eps)

def compute_flow(Q_m3s, D_m, L_m, rho, mu, eps, K_minor=0.0, elev_diff=0.0, Pin_pa=0.0):
    A = pi * (D_m**2) / 4.0
    V = Q_m3s / A if A > 0 else np.nan
    Re = rho * V * D_m / mu if mu > 0 else np.nan
    f = friction_factor(Re, D_m, eps)
    hf = f * (L_m / D_m) * (V**2 / (2.0 * g)) if D_m > 0 else np.nan
    h_minor = K_minor * (V**2 / (2.0 * g))
    h_total = hf + h_minor
    deltaP = rho * g * h_total
    Pout_pa = Pin_pa - deltaP - rho * g * elev_diff  # includes elevation difference
    return {
        "area_m2": A,
        "velocity_m_s": V,
        "Re": Re,
        "f": f,
        "hf_m": hf,
        "h_minor_m": h_minor,
        "h_total_m": h_total,
        "deltaP_Pa": deltaP,
        "Pout_Pa": Pout_pa
    }

st.set_page_config(page_title="Simulador de Escoamento - Cervejarias", layout="wide")

st.title("Simulador de Escoamento em Tubulações — Cervejaria")
st.markdown("Manipule vazão, diâmetro, pressão, altura, viscosidade e veja as variáveis interligadas (Reynolds, f, perda de carga, queda de pressão).")

# --- Inputs ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Geometria e Vazão")
    Q_val = st.number_input("Vazão (litros/min)", value=500.0, min_value=0.0, format="%.3f")
    D_mm = st.slider("Diâmetro (mm)", min_value=6.0, max_value=300.0, value=50.0, step=1.0)
    L_m = st.number_input("Comprimento da tubulação (m)", value=30.0, min_value=0.0)
    eps_mm = st.number_input("Rugosidade absoluta ε (mm)", value=0.015, min_value=0.0, format="%.6f")
    K_minor = st.number_input("Coeficiente de perdas singulares (K total)", value=0.0, format="%.4f")

with col2:
    st.subheader("Propriedades do fluido e pressão")
    rho = st.number_input("Densidade ρ (kg/m³)", value=1000.0)
    mu_cP = st.number_input("Viscosidade dinâmica μ (cP = mPa·s)", value=1.0, min_value=0.0001, format="%.6f")
    Pin_bar = st.number_input("Pressão de entrada (bar gauge)", value=1.0)
    elev_diff_m = st.number_input("Diferença de elevação (saida - entrada) m", value=0.0)

st.markdown("---")
st.write("Unidades: todas convertidas internamente para SI. cP -> Pa·s etc.")

# Conversions
Q_m3s = Q_val / 1000.0 / 60.0            # L/min -> m3/s
D_m = D_mm / 1000.0
eps_m = eps_mm / 1000.0
mu_Pa_s = mu_cP / 1000.0                 # cP to Pa.s
Pin_pa = (Pin_bar) * 1e5                 # bar gauge ~ bar abs approx (user can adjust)

# Compute
res = compute_flow(Q_m3s, D_m, L_m, rho, mu_Pa_s, eps_m, K_minor=K_minor, elev_diff=elev_diff_m, Pin_pa=Pin_pa)

# --- Outputs ---
col3, col4 = st.columns(2)
with col3:
    st.subheader("Resultados principais")
    st.metric("Velocidade (m/s)", f"{res['velocity_m_s']:.3f}")
    st.metric("Número de Reynolds", f"{res['Re']:.0f}")
    st.metric("Fator de atrito f", f"{res['f']:.5f}")

with col4:
    st.subheader("Perdas e Pressões")
    st.metric("Perda por atrito hf (m)", f"{res['hf_m']:.3f}")
    st.metric("Perda total (m c/ singulares)", f"{res['h_total_m']:.3f}")
    st.metric("Queda de pressão ΔP (kPa)", f"{res['deltaP_Pa']/1000.0:.3f}")
    st.metric("Pressão de saída (bar)", f"{res['Pout_Pa']/1e5:.4f}")

st.markdown("### Detalhes")
st.write({
    "Área (m²)": res["area_m2"],
    "Reynolds": res["Re"],
    "Fator f": res["f"],
    "hf (m)": res["hf_m"],
    "Perdas singulares (m)": res["h_minor_m"],
    "Perda total (m)": res["h_total_m"],
    "ΔP (Pa)": res["deltaP_Pa"],
    "P_out (Pa)": res["Pout_Pa"]
})

# Plot: velocidade profile / sensitivity (variação Q)
st.markdown("---")
st.subheader("Gráfico: Sensibilidade da perda de carga em função da vazão")
Q_range_lpm = np.linspace(max(0.001, Q_val*0.1), Q_val*2.0 + 1e-6, 100)
hf_list = []
for q in Q_range_lpm:
    q_m3s = q / 1000.0 / 60.0
    r = compute_flow(q_m3s, D_m, L_m, rho, mu_Pa_s, eps_m, K_minor=K_minor)
    hf_list.append(r['h_total_m'])

fig, ax = plt.subplots()
ax.plot(Q_range_lpm, hf_list)  # do not set color explicitly per notebook rules
ax.set_xlabel("Vazão (L/min)")
ax.set_ylabel("Perda total (m)")
ax.set_title("Perda total vs Vazão")
st.pyplot(fig)

st.markdown("### Observações")
st.markdown("""
- Mudanças na viscosidade (ex.: mostos densos) podem aumentar muito as perdas via Reynolds menor e fator distinto.
- Para tubos longos e depósitos elevados, verifique também a capacidade de bomba (NPSH, seleção de bomba não modelada aqui).
- Para linhas com múltiplas bombas, válvulas parcialmente fechadas ou transientes, este modelo steady-state é uma aproximação.
""")
