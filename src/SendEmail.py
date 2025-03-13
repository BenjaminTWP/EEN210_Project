import yagmail

def send_email(RECEIVER_EMAIL):

    SENDER_EMAIL = "EMAIL ADRESS OF SENDER"
    SENDER_PASSWORD = "YEA YOU KNOW"

    yag = yagmail.SMTP(SENDER_EMAIL, SENDER_PASSWORD)
    yag.send(
        to=RECEIVER_EMAIL,
        subject="Possible Fall Detected",
        contents="Hej du är kontaktperson till en person med falldetektor. Personen har troligtvis ramlat, skynda dig. Spring men ramla inte.")