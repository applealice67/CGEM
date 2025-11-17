from decimal import Decimal, InvalidOperation
import os

import re
import csv
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaModel, RobertaTokenizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ========== DEVICE SETUP ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== LOGGING ==========
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('log.txt'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def dual_output(message):
    print(message)
    logger.info(message)


def if_manufacturer_match(m1, m2, t1, t2):
    # 192 3199

    if m1 and m2:
        if m1 == m2:
            return 1
        else:
            return 0

    if not m1:
        if m2 in t1:
            return 1
        else:
            return -1

    if not m2:
        if m1 in t2:
            return 1
        else:
            return -1
    return -1


import re
from decimal import Decimal, InvalidOperation


def precise_abs_difference(input1, input2):
    def extract_number(s):
        """Extract the first valid number (integer/decimal/scientific notation) from a string"""

        if s is None or s == '':
            return None

        # Match numbers (including negatives, decimals, scientific notation)
        num_match = re.search(r'-?\d+\.?\d*(?:[eE][-+]?\d+)?', str(s))
        if num_match:
            try:
                return Decimal(num_match.group())
            except (InvalidOperation, TypeError):
                pass
        return None

    num1 = extract_number(input1)
    num2 = extract_number(input2)

    # Handle None or empty string cases
    if num1 is None and num2 is None:
        return 0  # Both invalid, treat as equal
    if num1 is None or num2 is None:
        return -1

        # Calculate the absolute difference and convert to integer (truncate decimals)
    difference = abs(num1 - num2)
    return int(difference)  # Or use round(difference, 0) for rounding


def common_words_after_removal(m1: str, m2: str, t1: str, t2: str) -> str:
    """Remove manufacturers from strings and return common words."""

    if m1 and m2:
        t1 = t1.replace(m1, '', 1)  # Remove m1 from t1
        t2 = t2.replace(m2, '', 1)  # Remove m2 from t2

    elif not m1:
        if m2 in t1:
            t1 = t1.replace(m2, '', 1)  # Remove m2 from t1
        t2 = t2.replace(m2, '', 1)  # Remove m2 from t2

    elif not m2:
        if m1 in t2:
            t2 = t2.replace(m1, '', 1)  # Remove m1 from t2
        t2 = t2.replace(m1, '', 1)  # Remove m1 from t2

    # Split t1 and t2 into word lists
    words1 = set(t1.split())
    words2 = set(t2.split())

    # Find common words between the two sets
    common_words_set = words1.intersection(words2)

    # Extract common words in the order of t1
    result = ' '.join([word for word in t1.split() if word in common_words_set])

    return t1, t2, result


def parse_line_todic(line):
    """Parse a single line of string into a dictionary"""
    result = {}
    tokens = line.split()

    key = None
    value = []

    for token in tokens:
        if token == "COL":
            if key:
                # Save the current field name and value
                if key in result:
                    if isinstance(result[key], tuple):
                        result[key] += (" ".join(value),)
                    else:
                        result[key] = (result[key], " ".join(value))
                else:
                    result[key] = " ".join(value)
            key = None
            value = []
        elif token == "VAL":
            key = key  # Ensure key has been assigned
        elif key is None:
            key = token  # Current token is the field name
        else:
            value.append(token)  # Current token is part of the value

    # Process the last field
    if key:
        if key in result:
            if isinstance(result[key], tuple):
                result[key] += (" ".join(value),)
            else:
                result[key] = " ".join(value)
        else:
            result[key] = " ".join(value)

    if value and value[-1].isdigit():
        result["label"] = int(value[-1])
        value.pop()  # Remove the last number
        # Update the last field's value
        if key in result:
            if isinstance(result[key], tuple):
                result[key] = result[key][:-1] + (" ".join(value),)
            else:
                result[key] = " ".join(value)
    return result


def parse_file_todic(file_path):
    """Read each line from a file and parse it into a dictionary"""
    final_result = {}
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()
    for line in lines:
        line = line.strip()  # Remove leading/trailing whitespace
        if not line:
            continue  # Skip empty lines
        parsed_line = parse_line_todic(line)

        # Merge the parsed result into the final dictionary
        for key, value in parsed_line.items():
            if key in final_result:
                final_result[key].append(value)
            else:
                final_result[key] = [value]
    return final_result


def dict_tolist(dic_data):
    keys = list(dic_data.keys())
    dual_output(keys)
    list_data = []
    for i in range(len(dic_data["label"])):
        item = []
        for key in keys:
            item.append(dic_data[key][i])
        list_data.append(item)
    return list_data, keys


def data_augment(list_data):
    """Perform data augmentation by balancing positive and negative samples."""
    result_data = []
    t_data = []
    f_data = []
    for row in list_data:
        if row[-1]:
            t_data.append(row)
        else:
            f_data.append(row)
    len_t = len(t_data)
    len_f = len(f_data)
    len_max = max(len_t, len_f)
    dual_output(f"t_len:{len_t}, f_len:{len_f}")
    for i in range(len_max):
        result_data.append(t_data[i % len_t])
        result_data.append(f_data[i % len_f])
        if i % 5 == 0:
            result_data.append(t_data[i % len_t])
    return result_data


import re

stop_words = set(
    ['such', "you'd", 'y', 't', 'down', 'i', 'by', 'whom', 'most', 'his', 'does', 'are', 'between', 're', 'isn', 'only',
     'she', 'of', 'had', 'through', 'other', 'needn', 'be', 'below', 'should', 'when', 'on', 'for', "don't", 'until',
     'can', 'to', 'a', 'from', 'has', "you'll", 'few', 'were', "that'll", 'while', 'just', "she's", "didn't", 'again',
     'under', 'him', 'these', 'your', 'this', 'that', 'being', 'doing', 'all', 'with', "haven't", 'didn', 'nor', 'they',
     'where', 'our', 'them', 'couldn', 'm', "needn't", 'me', 'you', 'we', 'than', "wouldn't", "shan't", 'ma', 'won',
     'yourselves', 'wouldn', 'haven', "it's", 'against', 'ain', 'have', 's', 'any', 'do', 'himself', 'there', 'what',
     'myself', 'both', 've', 'up', 'mustn', 'or', 'wasn', 'into', 'which', "shouldn't", 'hadn', 'as', 'own', 'o',
     'mightn', 'an', 'don', 'her', 'weren', 'itself', 'those', 'how', 'hers', "mightn't", 'is', 'was', "wasn't",
     'before', 'if', 'it', 'will', 'once', 'did', 'same', "hadn't", 'now', 'll', 'no', 'shan', "you're", 'too', 'aren',
     'he', 'some', 'my', 'over', "doesn't", 'shouldn', "isn't", 'ourselves', 'd', 'am', 'themselves', "aren't", 'off',
     'having', 'in', "hasn't", 'further', "mustn't", 'yourself', 'ours', 'theirs', 'here', 'more', 'so', "won't",
     'very', "should've", 'out', 'the', 'and', 'who', 'their', 'but', "couldn't", 'hasn', 'doesn', 'not', 'above',
     'because', 'about', 'its', 'during', "weren't", 'herself', 'been', 'yours', "you've", 'why', 'after', 'then',
     'each', 'at'])


def noStopwords(line):
    # Retain decimal points in numbers, replace other punctuation with spaces
    line = re.sub(r'(?<!\d)\.(?!\d)', ' ', line)  # Remove dots not between digits
    line = re.sub(r'[^\w\s\.]', ' ', line)  # Remove other non-alphanumeric characters
    line = re.sub(r'\s+', ' ', line).strip()  # Compress extra spaces

    tokens = line.lower().split()
    filtered_tokens = [w for w in tokens if w not in stop_words]
    return ' '.join(filtered_tokens)


def _process_AGdata_entry(title_pair, manufacturer_pair, price_pair, label):
    # {'title': 0.14652534375208545, 'manufacturer': 0.05769829852974523, 'price': 0.20776585468018058,
    t1, t2 = title_pair
    m1, m2 = manufacturer_pair
    p1, p2 = price_pair
    p = precise_abs_difference(p1, p2)

    s1 = "title " + t1 + " price " + str(p1) + " manufacture " + m1
    s2 = "title " + t2 + " price " + str(p2) + " manufacture " + m1
    s1 = noStopwords(s1)
    s2 = noStopwords(s2)
    return [(s1, s2), (t1, t2), (str(0), str(p)), (m1, m2), label]


def _process_Bdata_entry(name_pair, factory_pair, style_pair, abv_pair, label):
    # {'Beer_Name': 0.24430906805413774, 'Brew_Factory_Name': 0.28038944996494647, 'Style': -0.27087976005199366, 'ABV': 0.20007473887936267,
    n1, n2 = name_pair
    f1, f2 = factory_pair
    t1, t2 = style_pair
    a1, a2 = abv_pair

    p = precise_abs_difference(a1, a2)

    s1 = "Beer_Name " + n1 + " Brew_Factory_Name " + f1 + " ABV " + str(a1) + " Style " + t1
    s2 = "Beer_Name " + n2 + " Brew_Factory_Name " + f2 + " ABV " + str(a2) + " Style " + t2

    s1 = noStopwords(s1)
    s2 = noStopwords(s2)

    return [(s1, s2), (n1, n2), (f1, f2), (str(0), str(p)), (t1, t2), label]


def _process_WAdata_entry(title_pair, category_pair, brand_pair, modelno_pair, price_pair, label):
    # {'title': -0.01971571078298578, 'category': 0.045308931761990134, 'brand': -0.02751322714297252, 'modelno': 0.1398147006881398, 'price': 0.15020384295658573}
    t1, t2 = title_pair
    c1, c2 = category_pair
    b1, b2 = brand_pair
    m1, m2 = modelno_pair
    p1, p2 = price_pair
    p = precise_abs_difference(p1, p2)

    s1 = "price " + str(p1) + " modelno " + m1 + " category " + c1 + " title " + t1 + " brand " + b1
    s2 = "price " + str(p2) + " modelno " + m2 + " category " + c2 + " title " + t2 + " brand " + b2

    s1 = noStopwords(s1)
    s2 = noStopwords(s2)

    return [(s1, s2), (str(0), str(p)), (m1, m2), (c1, c2), (t1, t2), (b1, b2), label]


def _process_ABdata_entry(name_pair, description_pair, price_pair, label):
    # {'name': np.float64(0.12580924334638566), 'description': np.float64(-0.08685249301028268), 'price': np.float64(0.04595287378569339), 'overall': np.float64(0.12580924334638566)}
    n1, n2 = name_pair
    d1, d2 = description_pair
    p1, p2 = price_pair
    p = precise_abs_difference(p1, p2)

    s1 = "name " + n1 + " price " + str(p1) + " description " + d1
    s2 = "name " + n2 + " price " + str(p2) + " description " + d2

    s1 = noStopwords(s1)
    s2 = noStopwords(s2)

    return [(s1, s2), (n1, n2), (str(0), str(p)), (d1, d2), label]


def _process_IAdata_entry(Song_Name_pair, Artist_Name_pair, Album_Name_pair, Genre_pair, Price_pair, CopyRight_pair,
                          Time_pair, Released_pair, label):
    song1, song2 = Song_Name_pair
    artist1, artist2 = Artist_Name_pair
    album1, album2 = Album_Name_pair
    gen1, gen2 = Genre_pair
    p1, p2 = Price_pair
    c1, c2 = CopyRight_pair
    t1, t2 = Time_pair
    r1, r2 = Released_pair
    p = precise_abs_difference(p1, p2)

    s1 = "time " + t1 + " song name " + song1 + " album name " + album1 + " released " + r1 + " artist name " + artist1 + " price " + str(
        p1) + " copyright " + c1 + " genre " + gen1
    s2 = "time " + t2 + " song name " + song2 + " album name " + album2 + " released " + r2 + " artist name " + artist2 + " price " + str(
        p2) + " copyright " + c2 + " genre " + gen2

    s1 = noStopwords(s1)
    s2 = noStopwords(s2)

    return [(s1, s2), (t1, t2), (song1, song2), (album1, album2), (r1, r2), (artist1, artist2), (str(0), str(p)),
            (c1, c2), (gen1, gen2), label]


def _process_FZdata_entry(name_pair, addr_pair, city_pair, phone_pair, type_pair, class_pair, label):
    # {'name': 0.47258647660820996, 'addr': 0.25200868914159175, 'city': 0.0546193907431387, 'phone': 0.3766036095312342, 'type': 0.019489420037281442, 'class': 0.5555150228810763,
    n1, n2 = name_pair
    a1, a2 = addr_pair
    c1, c2 = city_pair
    p1, p2 = phone_pair
    t1, t2 = type_pair
    cl1, cl2 = class_pair

    s1 = "class " + cl1 + " name " + n1 + " phone " + p1 + " addr " + a1 + " city " + c1 + " type " + t1
    s2 = "class " + cl2 + " name " + n2 + " phone " + p2 + " addr " + a2 + " city " + c2 + " type " + t2

    s1 = noStopwords(s1)
    s2 = noStopwords(s2)

    return [(s1, s2), (c1, c2), (n1, n2), (p1, p2), (a1, a2), (cl1, cl2), (t1, t2), label]


def _process_DAdata_entry(title_pair, authors_pair, venue_pair, year_pair, label):
    t1, t2 = title_pair
    a1, a2 = authors_pair
    v1, v2 = venue_pair
    y1, y2 = year_pair
    y = precise_abs_difference(y1, y2)

    s1 = "title " + t1 + " year " + y1 + " authors " + a1 + " venue " + v1
    s2 = "title " + t2 + " year " + y2 + " authors " + a2 + " venue " + v2

    s1 = noStopwords(s1)
    s2 = noStopwords(s2)

    return [(s1, s2), (t1, t2), (a1, a2), (str(0), str(y)), (v1, v2), label]


def _process_DGdata_entry(title_pair, authors_pair, venue_pair, year_pair, label):
    t1, t2 = title_pair
    a1, a2 = authors_pair
    v1, v2 = venue_pair
    y1, y2 = year_pair
    y = precise_abs_difference(y1, y2)

    s1 = "authors " + a1 + " title " + t1 + " venue " + v1 + " year " + y1
    s2 = "authors " + a2 + " title " + t2 + " venue " + v2 + " year " + y2

    s1 = noStopwords(s1)
    s2 = noStopwords(s2)

    return [(s1, s2), (a1, a2), (t1, t2), (str(0), str(y)), (v1, v2), label]


def _process_Cdata_entry(content, label):
    # {'content': 0.0, 'label': 1}
    c1, c2 = content
    c1 = noStopwords(c1)
    c2 = noStopwords(c2)

    # Tokenize (assuming noStopwords returns a string that needs to be split by spaces)
    words1 = set(c1.split())
    words2 = set(c2.split())

    # Find common words
    common_words = words1 & words2
    c = " ".join(common_words)

    # Truncate to 255 characters
    s1 = c1[:255]
    s2 = c2[:255]

    return [(s1, s2), (c, c), label]


# ========== MODEL ==========
model_name = 'roberta-base'
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = RobertaModel.from_pretrained(model_name).to(device)

loss_fn = nn.BCEWithLogitsLoss()


class MoEGate(nn.Module):
    def __init__(self, input_dim, num_experts=4, k=1):
        super().__init__()
        self.experts = nn.ModuleList([
            nn.Linear(input_dim, 1) for _ in range(num_experts)
        ])
        self.gate = nn.Linear(input_dim, num_experts)
        self.k = k

    def forward(self, x):
        # x shape: (batch, input_dim)
        gate_scores = F.softmax(self.gate(x), dim=1)  # shape: (batch, num_experts)

        if self.k < len(self.experts):
            topk_vals, topk_idx = gate_scores.topk(self.k, dim=1)
            # Ensure mask is created on the same device as gate_scores
            mask = torch.zeros_like(gate_scores).scatter_(1, topk_idx, topk_vals)
            gate_scores = mask

        # The forward pass of all experts automatically inherits the device from x
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)  # shape: (batch, num_experts, 1)
        moe_output = (gate_scores.unsqueeze(2) * expert_outputs).sum(dim=1)  # shape: (batch, 1)
        return moe_output


class SentenceSimilarityDataset(Dataset):
    def __init__(self, data, max_len, tokenizer):
        self.max_length = max_len
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        label = item[-1]
        elements = item[:-1]

        encodings_list = []
        for element in elements:
            if isinstance(element, tuple):
                text1, text2 = element
                encodings = self.tokenizer(
                    text1, text2, padding='max_length', truncation=True,
                    max_length=self.max_length, return_tensors='pt', add_special_tokens=True)
            else:
                encodings = self.tokenizer(
                    element, padding='max_length', truncation=True,
                    max_length=self.max_length, return_tensors='pt', add_special_tokens=True)
            encodings_list.append(encodings)

        input_ids = [enc['input_ids'].flatten() for enc in encodings_list]
        token_type_ids = [enc.get('token_type_ids', torch.zeros_like(enc['input_ids'])).flatten() for enc in
                          encodings_list]
        attention_mask = [enc['attention_mask'].flatten() for enc in encodings_list]

        return {
            'input_ids': input_ids,
            'token_type_ids': token_type_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(label, dtype=torch.long)
        }


class WeightedClassifier(nn.Module):
    def __init__(self, hidden_size, num_chunks, learnable_alpha=True, use_moe=False):
        super(WeightedClassifier, self).__init__()

        self.num_chunks = num_chunks
        self.chunk_size = hidden_size // num_chunks
        self.concat_size = self.chunk_size * 2  # [x0; xi]

        if use_moe:
            self.gate_layers = nn.ModuleList([
                MoEGate(self.concat_size, num_experts=4, k=1) for _ in range(num_chunks - 1)
            ])
        else:
            self.gate_layers = nn.ModuleList([
                nn.Linear(self.concat_size, 1) for _ in range(num_chunks - 1)
            ])

        # Update classifier input dimension to (num_chunks - 1) * concat_size
        self.classifier = nn.Linear(self.concat_size * (num_chunks - 1), 1)

        if alpha is not None:
            assert len(alpha) == num_chunks - 1, "alpha length must be num_chunks - 1"
            self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float32), requires_grad=learnable_alpha)
        else:
            self.alpha = nn.Parameter(torch.ones(num_chunks - 1) / (num_chunks - 1), requires_grad=learnable_alpha)

        dual_output(f"self alpha {self.alpha.detach().cpu().numpy()}")

    def forward(self, x, return_gates=False):
        x_chunks = x.chunk(self.num_chunks, dim=1)
        x0 = x_chunks[0]

        gated_chunks = []
        gate_values = []

        alpha_raw = F.softmax(self.alpha, dim=0)  # shape: [num_chunks - 1]

        for i in range(1, self.num_chunks):
            xi = x_chunks[i]
            concat = torch.cat([x0, xi], dim=1)  # [B, 2 * chunk_size]

            gate_score = torch.sigmoid(self.gate_layers[i - 1](concat))  # [B, 1]
            gated = concat * gate_score * alpha_raw[i - 1]  # Scale the entire concatenated vector

            gated_chunks.append(gated)
            gate_values.append(gate_score.detach().cpu())

        concat_output = torch.cat(gated_chunks, dim=1)  # [B, (num_chunks - 1) * 2 * chunk_size]
        logits = self.classifier(concat_output)

        return (logits, gate_values) if return_gates else logits


class WeightedClassifier1(nn.Module):
    def __init__(self, hidden_size, num_chunks, learnable_alpha=True, use_moe=False):
        super(WeightedClassifier, self).__init__()

        self.num_chunks = num_chunks
        self.chunk_size = hidden_size // num_chunks

        # Start allocating gating layers from the second chunk
        if use_moe:
            self.gate_layers = nn.ModuleList([
                MoEGate(self.chunk_size, num_experts=4, k=1) for _ in range(num_chunks - 1)
            ])
        else:
            self.gate_layers = nn.ModuleList([
                nn.Linear(self.chunk_size, 1) for _ in range(num_chunks - 1)
            ])

        self.classifier = nn.Linear(hidden_size, 1)

        # Alpha parameter: passed for all chunks, including the first one
        if alpha is not None:
            assert len(alpha) == num_chunks, "Length of alpha must equal num_chunks"
            self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float32), requires_grad=learnable_alpha)
        else:
            self.alpha = nn.Parameter(torch.ones(num_chunks) / num_chunks, requires_grad=learnable_alpha)

        dual_output(f"self alpha {self.alpha.detach().cpu().numpy()}")

    def forward(self, x, return_gates=False):
        x_chunks = x.chunk(self.num_chunks, dim=1)

        gated_chunks, gate_values = [], []

        alpha_raw = F.softmax(self.alpha, dim=0)  # Normalize

        for i, chunk in enumerate(x_chunks):
            if i == 0:
                # First chunk: only multiply by alpha, no gating
                gated_chunk = chunk * alpha_raw[0]
                gate_values.append(torch.ones_like(chunk[:, :1]))  # Mark as pass-through
            else:
                gate_score = torch.sigmoid(self.gate_layers[i - 1](chunk))  # Index from gate_layers[0]
                weighted_gate = gate_score * alpha_raw[i]
                gated_chunk = chunk * weighted_gate
                gate_values.append(gate_score.detach().cpu())

            gated_chunks.append(gated_chunk)

        concat_output = torch.cat(gated_chunks, dim=1)
        logits = self.classifier(concat_output)

        return (logits, gate_values) if return_gates else logits


from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup


def train(trainData, testData, first_load=False):
    train_dataset = SentenceSimilarityDataset(trainData, max_len, tokenizer)
    test_dataset = SentenceSimilarityDataset(testData, max_len, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    classifier = WeightedClassifier(model.config.hidden_size * pairNum, pairNum).to(device)

    # Use AdamW optimizer
    optimizer = AdamW([
        {'params': model.parameters(), 'lr': 3e-5},
        {'params': classifier.gate_layers.parameters(), 'lr': 1e-4},
        {'params': classifier.classifier.parameters(), 'lr': 1e-4},
        {'params': [classifier.alpha], 'lr': 1e-5}
    ], weight_decay=1e-4)

    # Set scheduler: 10% warmup, then linear decay
    total_steps = len(train_loader) * num_epochs
    warmup_steps = int(0.1 * total_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    best_f1 = 0
    best_epoch = 0

    if first_load and os.path.exists(save_path):
        checkpoint = torch.load(save_path, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        classifier.load_state_dict(checkpoint['classifier_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint.get('scheduler_state_dict', {}))
        best_f1 = checkpoint.get('best_f1', 0)
        best_epoch = checkpoint.get('epoch', 0)
        dual_output(f"Loaded model from {save_path}, best F1: {best_f1:.4f} at epoch {best_epoch + 1}")

    for epoch in range(num_epochs):
        model.train()
        classifier.train()
        train_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            input_ids_list = [x.to(device) for x in batch['input_ids']]
            attention_mask_list = [x.to(device) for x in batch['attention_mask']]
            labels = batch['labels'].view(-1, 1).float().to(device)

            outputs_list = []
            for input_ids, attention_mask in zip(input_ids_list, attention_mask_list):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                outputs_list.append(outputs.last_hidden_state[:, 0, :])

            combined_outputs = torch.cat(outputs_list, dim=1)
            logits = classifier(combined_outputs)

            loss = loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()  # Update learning rate each step

            train_loss += loss.item()

            if batch_idx % 50 == 0:
                dual_output(
                    f"Epoch {epoch + 1}, Batch {batch_idx}, loss: {loss.item():.4f}, total_loss: {train_loss:.4f}")

        model.eval()
        classifier.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in test_loader:
                input_ids_list = [x.to(device) for x in batch['input_ids']]
                attention_mask_list = [x.to(device) for x in batch['attention_mask']]
                labels = batch['labels'].to(device)

                outputs_list = []
                for input_ids, attention_mask in zip(input_ids_list, attention_mask_list):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    outputs_list.append(outputs.last_hidden_state[:, 0, :])

                combined_outputs = torch.cat(outputs_list, dim=1)
                logits = classifier(combined_outputs)
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).long().flatten()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)

        dual_output(
            f"Epoch {epoch + 1} Test Metrics: Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

        if hasattr(classifier, "alpha"):
            alpha_now = torch.softmax(classifier.alpha, dim=0).detach().cpu().numpy()
            alpha_str = ', '.join([f"{a:.3f}" for a in alpha_now])
            dual_output(f"Alpha weights (importance per chunk): [{alpha_str}]")

        if hasattr(classifier, "gate_layers"):
            classifier.eval()
            with torch.no_grad():

                dummy_input = torch.randn(1, classifier.chunk_size).to(device)

                gate_means = []
                for i, gate_layer in enumerate(classifier.gate_layers):
                    in_dim = gate_layer.in_features  # Linear layer attributes
                    dummy_input = torch.randn(1, in_dim, device=device)
                    g = torch.sigmoid(gate_layer(dummy_input)).item()
                    gate_means.append(g)
                gate_str = ', '.join([f"G{i + 1}={g:.3f}" for i, g in enumerate(gate_means)])
                dual_output(f"Gate preview (single input): [{gate_str}]")

        if f1 > best_f1:
            torch.save({
                'model_state_dict': model.state_dict(),
                'classifier_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch,
                'best_f1': f1
            }, save_path)
            best_f1 = f1
            best_epoch = epoch
            dual_output(f"Saved best model with F1 {f1:.4f}")
        else:
            checkpoint = torch.load(save_path, weights_only=True)
            model.load_state_dict(checkpoint['model_state_dict'])
            classifier.load_state_dict(checkpoint['classifier_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint.get('scheduler_state_dict', {}))
            best_f1 = checkpoint.get('best_f1', 0)
            best_epoch = checkpoint.get('epoch', 0)
            dual_output(f"Loaded model best F1: {best_f1:.4f} at epoch {best_epoch + 1}")

    dual_output(f"Training complete. Best F1: {best_f1:.4f} at epoch {best_epoch + 1}")

    # ========== Load best model and write prediction results ==========
    checkpoint = torch.load(save_path, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    classifier.load_state_dict(checkpoint['classifier_state_dict'])

    model.eval()
    classifier.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids_list = [x.to(device) for x in batch['input_ids']]
            attention_mask_list = [x.to(device) for x in batch['attention_mask']]
            labels = batch['labels'].to(device)

            outputs_list = []
            for input_ids, attention_mask in zip(input_ids_list, attention_mask_list):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                outputs_list.append(outputs.last_hidden_state[:, 0, :])

            combined_outputs = torch.cat(outputs_list, dim=1)
            logits = classifier(combined_outputs)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).long().flatten()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # ========== Write results to result.csv ==========
    import csv
    result_path = "result.csv"  # Can customize the path
    with open(result_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Index", "True Label", "Predicted Label"])
        for idx, (true, pred) in enumerate(zip(all_labels, all_preds)):
            writer.writerow([idx, true, pred])

    dual_output(f"Prediction results saved to {result_path}")


import argparse

# ========== MAIN ==========
if __name__ == "__main__":
    max_len = 256
    firstload = False

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Company')
    # 'Amazon-Google', 'Beer', 'Fodors-Zagats', 'Walmart-Amazon', 'Abt-Buy', "iTunes-Amazon", "DBLP-ACM", "DBLP-GoogleScholar", "Company"
    args = parser.parse_args()
    dataset = args.dataset

    if dataset == "Amazon-Google":
        # alpha=[0.1426, 0.1388, 0.2607, 0.0532]
        # alpha=[1.0, 1.0, 2.0, 0.0]
        alpha = [1.0, 2.0, 0.0]
        path = 'data/er_magellan/Structured'
        batch_size = 16
        num_epochs = 30

    elif dataset == 'Beer':
        # alpha=[0.2727, 0.2970, 0.3068, 0.1748, -0.2054]
        # alpha=[1.0, 1.0, 1.0, 1.0, 0.0]
        # alpha=[2.0, 2.0, 1.0, 0.0]
        alpha = [1.0, 1.0, 1.0, 0.0]
        path = 'data/er_magellan/Structured'
        batch_size = 8
        num_epochs = 50

    elif dataset == 'Fodors-Zagats':
        alpha = [1.0, 1.0, 1.0, 1.0, 0.0, 0.0]
        path = 'data/er_magellan/Structured'
        batch_size = 16
        num_epochs = 20

    elif dataset == 'Walmart-Amazon':
        # alpha=[0.1664, 0.1925, 0.1437, 0.0843, 0.0470, -0.0622]
        # alpha=[4.0, 3.0, 2.0, 1.0, 0.0]
        # alpha=[2.0, 2.0, 1.0, 1.0, 0.0]
        alpha = [1.0, 1.0, 1.0, 0.0, 0.0]
        path = 'data/er_magellan/Dirty'
        batch_size = 16
        num_epochs = 20

    elif dataset == 'Abt-Buy':
        # alpha=[0.1301, 0.1968, 0.0274, -0.0091]
        # alpha=[1.0, 1.0, 1.0, 0.0]
        alpha = [1.0, 1.0, 1.0]  # Best
        # alpha=[2.0, 1.0, 0.0]
        path = 'data/er_magellan/Textual'
        batch_size = 16
        num_epochs = 20

    elif dataset == "iTunes-Amazon":
        # alpha=[0.2911, 0.6661, 0.5724, 0.2905, 0.2834, 0.1429, 0.1, 0.0440, -0.2014]
        # alpha=[2.0, 3.0, 3.0, 2.0, 2.0, 1.0, 1.0, 1.0, 0.0]
        alpha = [3.0, 3.0, 2.0, 2.0, 1.0, 1.0, 1.0, 0.0]
        path = 'data/er_magellan/Dirty'
        batch_size = 8
        num_epochs = 20

    elif dataset == "DBLP-ACM":
        # alpha=[1.0, 1.0, 1.0, 1.0, 1.0]
        # alpha=[0.5942, 0.6081, 0.6022, 0.5832, 0.1003]
        alpha = [1.0, 1.0, 1.0, 1.0]
        path = 'data/er_magellan/Dirty'
        batch_size = 16
        num_epochs = 20

    elif dataset == "DBLP-GoogleScholar":
        # alpha=[0.4399, 0.5269, 0.2591, 0.0152]
        # alpha=[1.0, 1.0, 1.0, 1.0, 0.0]
        alpha = [1.0, 1.0, 1.0, 0.0]
        path = 'data/er_magellan/Dirty'
        batch_size = 16
        num_epochs = 20

    elif dataset == "Company":
        alpha = [1.0]
        path = 'data/er_magellan/Textual'
        batch_size = 16
        num_epochs = 20

    save_path = f'{path}/{dataset}/best_model.pth'
    gate_dir = f'{path}/{dataset}/'

    trainData, testData, pairNum = getData(path, dataset)

    dual_output("Loading trainData and testData")
    dual_output("Starting training...")
    train(trainData, testData, first_load=firstload)
    dual_output("Process completed.")
