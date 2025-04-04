import os

user_input = input("Enter a filename to list: ")
os.system(f"ls {user_input}")  # 🔥 Vulnerable to command injection!
