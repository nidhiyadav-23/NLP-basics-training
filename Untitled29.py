#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install SpeechRecognition pyttsx3 pyaudio


# In[ ]:





# In[1]:


import pyttsx3

# Initialize the speech engine
engine = pyttsx3.init()

# Function to speak the text
def speak(text):
    engine.say(text)
    engine.runAndWait()

# Main function
def main():
    speak("Hello, I am your simple bot from Mallareddy College.")
    speak("You can say 'hello' or ask my name.")
   
    while True:
        command = input("You: ").lower()
       
        if "hello" in command:
            speak("Hey there!")
        elif "what's your name" in command or "your name" in command:
            speak("My name is Mallareddy Bot.")
        elif "goodbye" in command or "exit" in command:
            speak("Goodbye! Have a great day!")
            break
        else:
            speak("I didn't understand that. Please try again.")

# Entry point
if __name__ == "__main__":
    main()



# In[2]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

years=[[2015],[2016],[2017],[2018]]
values=[200, 220, 260, 280]

years_train, years_test, values_train, values_test = train_test_split(years, values, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(years_train, values_train)

future_years = [[2019]]
predicted_value = model.predict(future_years)
print(f"Predicted values for future years {future_years[0][0]}: {predicted_value[0]:.2f}")


# In[ ]:




