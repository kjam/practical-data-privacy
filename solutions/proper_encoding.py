hasher = blake2b()
hasher.update(username.encode('utf-8'))
hasher.hexdigest()
