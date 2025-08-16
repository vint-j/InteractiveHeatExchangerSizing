# Double-Pipe Heat Exchanger + TCU (Chiller + Heater) Interactive App

An interactive **Streamlit** app for sizing and simulating a double-pipe heat exchanger with a temperature control unit (TCU).  
👉 Try it live **[here](https://vint-j-interactiveheatexchangersizing-script-vmhpea.streamlit.app/)**

- Models both **cooling** (via chiller supply) and **heating** (via electric heater with power limit).  
- Supports **manual inputs + sliders** (synced).  
- Estimates **required HX length**, **UA**, **installed duty**, and **dynamic time to setpoint** based on loop volume and heat loads.  
- Highlights limitations (laminar flow, heater power cap, unreachable setpoints, etc).

---

## 🚀 Features
- Select pipe sizes from ASME B36.10 Sch 40 dimensions.
- Choose wall material and fouling resistances.
- Enter machine count and duty (or override total kW).
- Define flows, current process temp, setpoint, and loop volume.
- Chiller supply setpoint for cooling.
- Heater maximum power and target HTF supply for heating.
- **Dynamic solver** for time-to-setpoint (Euler integration).
- Clear HTML-style report with notes on validity and warnings.

---

## 🛠️ Requirements
- Python 3.9+  
- Install dependencies:

```bash
pip install streamlit
