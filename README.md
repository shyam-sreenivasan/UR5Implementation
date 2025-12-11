# UR5 Robot Simulator

Interactive 3D visualization of UR5 robot with trajectory planning.

## Quick Setup

### 1. Install Node.js
Download and install from: https://nodejs.org/
(Choose the LTS version)

Verify installation:
```bash
node --version
npm --version
```

### 2. Switch to Project
```bash
cd robot-viz
```

### 3. Install Three.js
```bash
npm install three
npm install @types/three --save-dev
```

### 6. Run
```bash
npm start
```

The app will open in your browser at `http://localhost:3000`

## Usage

- **Sliders**: Control each joint angle
- **Reset**: Return to home position
- **Random**: Random configuration
- **Select trajectory type**: Circle, Square, or Sine Wave
- **Init**: Generate trajectory path
- **Animate**: Execute the trajectory
- **Camera controls**: Top-right panel to adjust view
