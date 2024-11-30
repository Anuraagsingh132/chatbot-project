import tkinter as tk
from tkinter import scrolledtext, messagebox
import chatbot

fullscreen = False

def toggle_fullscreen(event=None):
    global fullscreen
    fullscreen = not fullscreen
    root.attributes("-fullscreen", fullscreen)

def exit_fullscreen(event=None):
    global fullscreen
    fullscreen = False
    root.attributes("-fullscreen", False)

def on_send(event=None):
    user_input = entry.get()
    if user_input.lower() == "quit":
        root.quit()
    else:
        chat_window.config(state=tk.NORMAL)
        chat_window.insert(tk.END, "You: " + user_input + '\n')
        chat_window.yview(tk.END)
        response = chatbot.chatbot_response(user_input)
        chat_window.insert(tk.END, "Bot: " + response + '\n')
        chat_window.yview(tk.END)
        chat_window.config(state=tk.DISABLED)
        entry.delete(0, tk.END)
    entry.focus()

def show_questions():
    try:
        with open('questions.txt', 'r') as file:
            questions = file.read()
        chat_window.config(state=tk.NORMAL)
        chat_window.insert(tk.END, "Bot: Here are some questions you can ask:\n" + questions + '\n')
        chat_window.yview(tk.END)
        chat_window.config(state=tk.DISABLED)
    except FileNotFoundError:
        messagebox.showerror("Error", "The questions.txt file was not found!")

def clear_all():
    chat_window.config(state=tk.NORMAL)
    chat_window.delete(1.0, tk.END)
    chat_window.config(state=tk.DISABLED)
    entry.delete(0, tk.END)

root = tk.Tk()
root.title("Enhanced Chatbot")
root.config(bg="#1C2833")

frame = tk.Frame(root, bg="#1C2833")
frame.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)

chat_window = scrolledtext.ScrolledText(frame, wrap=tk.WORD, font=("Arial", 14), state=tk.DISABLED, bg="#2E4053", fg="white", bd=0, relief="flat")
chat_window.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

entry_frame = tk.Frame(root, bg="#1C2833")
entry_frame.pack(pady=10, padx=10, fill=tk.X)

entry = tk.Entry(entry_frame, width=100, font=("Arial", 16), bd=2, relief="flat", fg="black", bg="#D5DBDB", insertbackground="black", highlightthickness=2, highlightbackground="#E74C3C")
entry.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.X, expand=True)
entry.insert(tk.END, "Type your message...")

def clear_placeholder(event):
    if entry.get() == "Type your message...":
        entry.delete(0, tk.END)
        entry.config(fg="black")

def restore_placeholder(event):
    if entry.get() == "":
        entry.insert(tk.END, "Type your message...")
        entry.config(fg="#BDC3C7")

entry.bind("<FocusIn>", clear_placeholder)
entry.bind("<FocusOut>", restore_placeholder)

send_button = tk.Button(entry_frame, text="Send", font=("Arial", 14), command=on_send, bd=2, relief="raised", bg="#28B463", fg="white", activebackground="#239B56")
send_button.pack(side=tk.RIGHT, padx=10, pady=5)

show_button = tk.Button(root, text="Show Questions", font=("Arial", 14), command=show_questions, bd=2, relief="raised", bg="#3498DB", fg="white", activebackground="#2980B9")
show_button.pack(pady=5)

clear_button = tk.Button(root, text="Clear", font=("Arial", 14), command=clear_all, bd=2, relief="raised", bg="#F1C40F", fg="black", activebackground="#D4AC0D")
clear_button.pack(pady=5)

entry.bind("<Return>", on_send)

label = tk.Label(root, text="Stress Management and Mental Health Support Bot", font=("Arial", 10), fg="white", bg="#1C2833")
label.pack(side=tk.BOTTOM, pady=5)

root.bind("<F11>", toggle_fullscreen)
root.bind("<Escape>", exit_fullscreen)

root.mainloop()
se