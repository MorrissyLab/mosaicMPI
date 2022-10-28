

def newline_wrap(string, length=40):
    return '\n'.join(string[i:i+length] for i in range(0, len(string), length))