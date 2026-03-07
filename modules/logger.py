import os
from datetime import datetime

class DataLogger:
    def __init__(self, logs_dir="logs", date_format="%Y-%m-%d", time_format="%H:%M:%S"):
        """
        Initializes the logger and creates the logs directory if it doesn't exist.
        """
        self.logs_dir = logs_dir
        self.date_format = date_format
        self.time_format = time_format
        
        if not os.path.exists(self.logs_dir):
            os.makedirs(self.logs_dir)
            
        self.current_date = datetime.now().strftime(self.date_format)
        self.log_file = os.path.join(self.logs_dir, f"{self.current_date}_cell_data.txt")
        
        if not os.path.exists(self.log_file):
            with open(self.log_file, "w") as f:
                f.write("Time,Frame,Total_Cells,Moving,Staying,Toxicity,Temperature\n")

    def log_counts(self, frame_count, counts, raw_detected, toxicity, temperature):
        timestamp = datetime.now().strftime(self.time_format)
        survivor_count = len(counts) if isinstance(counts, dict) else 0 
        
        log_entry = (f"{timestamp},{frame_count},{survivor_count},"
                    f"{raw_detected},{toxicity},{temperature}\n")
        
        try:
            with open(self.log_file, "a") as f:
                f.write(log_entry)
        except Exception as e:
            print(f"Logging error: {e}")