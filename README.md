# JARS - Jamming Analysis and Radio Simulation

This application simulates radio communication scenarios involving a transmitter, jammer, and receiver. It calculates the Jammer-to-Signal (J/S) ratio and visualizes the geometry of the setup.

## 🧱 Project Structure (MVC)

```
mvc/
├── model.py        # Simulation logic and domain classes (Model)
├── controller.py   # Coordination and calculation logic (Controller)
├── view.py         # Tkinter GUI and plotting logic (View)
```

## 🚀 Getting Started

### Prerequisites

* Python 3.8+
* Required packages - see pyproject.toml

```bash
pip install matplotlib
```

### Running the Application

From the root of the project:

```bash
python mvc/view.py
```

## 🧼 Features

* Input configuration for transmitter, jammer, and receiver.
* Calculates received powers and J/S ratio.
* Determines jamming success/failure against a threshold.
* Plots the geometry of the scenario in 2D with annotations.

## 📦 Extensibility

* Modular MVC structure for easy expansion.
* Add new source types or propagation models by extending `model.py`.
* Additional plots or simulation controls can be added in `view.py`.

