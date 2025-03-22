#utf-8 byte pair encoding

class Tokenizer:
    def __init__(self, vocabSize: int) -> None:
        self.vocabSize = vocabSize
        self.numMerges = vocabSize - 256
        self.merges = {}
        self.vocab = {}
    def merge(self, par: list, pair: list, tok: int) -> list:
        new = []
        i = 0
        length = len(par)
        while i < length-1:
            if par[i] == pair[0] and par[i+1] == pair[1]:
                new.append(tok)
                i += 2
            else:
                new.append(par[i])
                i += 1
        if i < length: new.append(par[length-1])
        return new
    def getStats(self, tokens: list) -> dict:
        count = {}
        for pair in zip(tokens, tokens[1:]): count[pair] = count.get(pair,0) + 1
        return count
    def train(self, trainText: str) -> None:
        trainTextd = trainText.encode("utf-8")
        tokens = list(map(int, trainTextd))
        merges = {}
        for i in range(self.numMerges):
            stats = self.getStats(tokens)
            pair = max(stats, key=lambda k: stats.get(k, 0))
            id = 256+i
            print(f"[+] {pair} -> {id}")
            tokens = self.merge(tokens, pair, id)
            merges[pair] = id
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in merges.items(): vocab[idx] = vocab[p0] + vocab[p1]
        self.merges = merges
        self.vocab = vocab
    def encode(self, text: str) -> list:
        tokens = list(map(int, text.encode("utf-8")))
        while True:
            pairs = self.getStats(tokens)
            pair = max(pairs)
            if pair not in self.merges: break
            tokens = self.merge(tokens, pair, self.merges[pair])
        return tokens
    def decode(self, tokens: list) -> str:
        tokensd = b"".join(self.vocab[idx] for idx in tokens)
        text = tokensd.decode("utf-8", errors="replace")
        return text
trainText = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Phasellus in mi mauris. Phasellus consectetur risus sem, et varius metus cursus tincidunt. Fusce pellentesque purus felis, quis tincidunt velit egestas eu. Pellentesque sagittis odio ut turpis volutpat, fringilla feugiat orci lacinia. Duis ut vehicula urna. Integer porttitor, massa ut consectetur maximus, lacus mauris gravida metus, eu consequat est ligula ut eros. Curabitur non malesuada velit, tempus aliquam mauris. Vestibulum leo turpis, tempus non dapibus et, fringilla eget augue. Sed tempus iaculis condimentum. Aenean non lacinia libero, vel lacinia ex. Nam venenatis metus in neque mollis dignissim."

tokenizer = Tokenizer(300)
tokenizer.train(trainText)
print(tokenizer.decode(tokenizer.encode("Curabitur non malesuada")))
