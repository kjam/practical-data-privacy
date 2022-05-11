def add_padding_and_encrypt(cipher, username):
    if len(username) < 4:
        username += "X" * (4-len(username))
    return cipher.encrypt(username)
