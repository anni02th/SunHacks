import datetime

class ClimateControlSystem:
    def _init_(self):
        self.ac_status = "OFF"

    def is_office_hours(self):
        now = datetime.datetime.now()
        # Check if it's office hours (Monday to Friday, 9:00 AM to 5:00 PM)
        if now.weekday() in range(0, 5) and datetime.time(9, 0) <= now.time() < datetime.time(17, 0):
            return True
        else:
            return False

    def is_indian_holiday(self):
        # Check if it's an Indian holiday (you need to define a function to check Indian holidays)
        # For simplicity, assume it's not a holiday for now
        return False

    def adjust_ac_settings(self):
        if self.is_office_hours() and not self.is_indian_holiday():
            self.ac_status = "ON"
        else:
            self.ac_status = "OFF"

    def display_ac_status(self):
        print("AC Status:", self.ac_status)

# Example usage
if __name__ == "_main_":
    climate_system = ClimateControlSystem()

    # Adjust AC settings based on office hours and holidays
    climate_system.adjust_ac_settings()
    climate_system.display_ac_status()