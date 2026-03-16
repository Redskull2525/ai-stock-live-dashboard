import schedule
import time
import os

def retrain():

    print("Retraining model...")

    os.system("python train_model.py")

schedule.every().day.at("22:00").do(retrain)

while True:

    schedule.run_pending()

    time.sleep(60)
