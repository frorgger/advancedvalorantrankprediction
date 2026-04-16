from ultralytics import solutions

def streamlit():
    solutions.inference(model="runs/detect/valorant_reaction_v220/weights/best.pt")

if __name__ == "__main__":
    streamlit()