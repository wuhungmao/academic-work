alphabet = "abcdefghijklmnopqrstuvwxyz"
ai = {c: i for i, c in enumerate(alphabet)}

ciphertext = "hfvkedyltegltea sa wmza vbwfkqsa ujcixrg hpxlru"
key = "friend"

def dvigenere(ct: str, key_text: str) -> str:
    key_text = "".join(ch for ch in key_text if ch in ai)
    result_chars = []
    key_index = 0
    key_len = len(key_text)
    for ch in ct:
        if ch not in ai:
            result_chars.append(ch)
            continue
        shift = ai[key_text[key_index % key_len]]
        p = (ai[ch] - shift) % 26
        result_chars.append(alphabet[p])
        key_index += 1
    return "".join(result_chars)


if __name__ == "__main__":
    plaintext = dvigenere(ciphertext, key)
    print(plaintext)


