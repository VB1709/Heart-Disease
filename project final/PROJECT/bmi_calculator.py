def calculate_bmi(weight_kg, height_m):
    """
    Calculate Body Mass Index (BMI) using weight in kilograms (kg) and height in meters (m).
    Formula: BMI = weight / (height^2)
    """
    if height_m <= 0:
        raise ValueError("Height must be a positive number.")

    bmi = weight_kg / (height_m ** 2)
    return bmi

def bmi_category(bmi):
    """
    Determine BMI category based on BMI value.
    """
    if bmi < 18.5:
        return "Underweight"
    elif 18.5 <= bmi < 25:
        return "Normal weight"
    elif 25 <= bmi < 30:
        return "Overweight"
    else:
        return "Obesity"


 

    
    
    