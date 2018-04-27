import json


class ModelConfiguration:
    def __init__(self, filename):
        with open(filename, encoding='utf8') as f:
            s = f.read()
        d = json.loads(s)
        for k, v in d.items():
            if k == 'emb_dim':
                self.emb_dim = v
            elif k == 'encoder_arch':
                self.encoder_arch = v
            elif k == 'decoder_arch':
                self.decoder_arch = v
            elif k == 'out_emb_dim':
                self.out_emb_dim = v
            elif k == 'dropout_rate':
                self.dropout_rate = v

    def to_json(self):
        return json.dumps(vars(self))
