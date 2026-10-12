# CartPole DQN

1. Open a terminal in the project folder, the folder that contains `train.py`.

2. Create and activate a Python virtual environment.

   On macOS or Linux:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

   On Windows PowerShell:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Run the training. It will run locally on CPU.

   ```bash
   python train.py
   ```

5. When training finishes, the results will be saved in a new folder under `outputs`, named `run_YYYYMMDD_HHMMSS`. This folder contains the trained model, charts, CSV files, and a results summary.

6. Open a window and watch the trained agent balance the pole for five episodes:

   ```bash
   python watch.py
   ```

   This uses the model from the newest run. Run it on a computer with a desktop display so the game window can open. Press Ctrl+C to stop early.
