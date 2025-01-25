# EEN210

# To run the code
1. Run the following lines to download required packages:
    - python -m venv .venv  # "py -3.11 -m venv .venv" #python >3.12 does not work
    - for ***Mac*** and ***Linux***:  
        - source .venv/bin/activate  
    - for ***Windows***:  
        - run terminal(vscode) with administrative privilage  
        - Set-ExecutionPolicy RemoteSigned  
        - .venv/Scripts/activate  # ".\.venv\Scripts\activate" if using PowerShell
    - pip install -r requirements.txt  
2. You need to change ***"Your_IP_Address"*** in index.html file;  
3. You need to also add ***"your WiFi SSID"***, ***"your Passoword"*** and ***"your WiFi IP Address"*** in main.cpp if you want to programm the micro controler.  
