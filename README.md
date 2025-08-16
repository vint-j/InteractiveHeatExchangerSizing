# Double-Pipe Heat Exchanger + TCU (Chiller + Heater) Interactive App

An interactive **Streamlit** app for sizing and simulating a double-pipe heat exchanger with a temperature control unit (TCU).  

You can try the hosted version **[here](https://vint-j-interactiveheatexchangersizing-script-vmhpea.streamlit.app/)**.

The app:
- Models both cooling (via chiller supply) and heating (via electric heater with a power limit).  
- Provides both manual input fields and sliders.  
- Estimates required HX length, UA, installed duty, and dynamic time to setpoint based on loop volume and heat loads.  
- Highlights limitations (laminar flow, heater power cap, unreachable setpoints, etc).

---

## Features
- Pipe sizes from ASME B36.10 Sch 40 dimensions.
- Select wall material and fouling resistances.
- Define machine count and duty (with optional override total kW).
- Specify flows, process temperatures, setpoint, and loop volume.
- Chiller supply setpoint for cooling mode.
- Heater maximum power and target HTF supply for heating mode.
- Dynamic solver for time-to-setpoint (Euler integration).
- Generates a clear technical report with warnings and notes.

---

## Running Locally
Requirements (only needed if you want to run this app locally instead of using the hosted link):  

- Python 3.9+  
- Streamlit  

Install with:  
```bash
pip install streamlit
