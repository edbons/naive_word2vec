import numpy as np
from collections import Counter
from time import time
from argparse import ArgumentParser
import pickle
import re
import logging

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)


class NaiveWord2VecNGS:
    """Naive implementation of Word2Vec model with negative sampling."""

    def __init__(self, window_size=5, learning_rate=0.05, n_dim=100, epochs=5, neg_samples=5, save_output=False) -> None:
        """
        Args:
            window_size (int): Count of context words before and after center word. Total count of center words = window_size * 2. Default: 5
            verbose (bool, optional): Print some statistics. Default: False          
            learning_rate (float): Learning rate coefficient for gradient descent optimizer. Default: 0.05
            n_dim (int): Size of word vector. Default: 100
            epochs (int): Number of epochs to train model. Default: 5
            neg_samples (int): Number of negative samples. Default: 5
        """
        self.idx2word = None
        self.word2idx = None
        self.window_size = window_size
        self.W_emb = None
        self.W_dense = None
        self.gradients = {}
        self.parameters = {}
        self.learning_rate = learning_rate
        self.w2v = {}  #: Dict with a center word as a key, and word vector as value 
        self.n_dim = n_dim
        self.epochs = epochs
        self.word_dist = None
        self.neg_samples = neg_samples
        self.save_output = save_output
       
    def _build_vocab(self, text,  unknown_tag='<UNK>'):
        self.idx2word = [unknown_tag] + sorted(set(text.split()))
        self.word2idx = {word: i for i, word in enumerate(self.idx2word)}
        self.vocab_size = len(self.idx2word)

    def _noise_dist(self, sentence):
        alpha      = 3 / 4
        noise_dist = {k: (v / len(sentence)) ** alpha for k, v in Counter(sentence).items()}
        Z = sum(noise_dist.values())
        self.word_dist = {self.word2idx[key]: val / Z for key, val in noise_dist.items()}

    def _preprocces_text(self, text, unknown_tag='<UNK>'):
        cwords = []
        owords = []
        sentence = text.split()

        self._noise_dist(sentence)
        
        for i, word in enumerate(sentence):
            cwords.append(self.word2idx[word])
            left = sentence[max(i-self.window_size, 0):i]
            right = sentence[i +
                             1:min(i+self.window_size, len(sentence)-i) + 1]
            others = left + right + [unknown_tag] * (abs(2 * self.window_size-(len(
                right) + len(left))) if (len(right) + len(left)) < self.window_size * 2 else 0)
            owords.append([self.word2idx[word] for word in others])

        x = np.array(cwords, dtype=np.uint32)
        x = np.expand_dims(x, axis=1)

        context_idxs = np.array(owords, dtype=np.uint32)

        return x, context_idxs

    def _init_parameters(self):
        self.W_emb = np.random.uniform(
            0, 1, size=(self.vocab_size, self.n_dim)) * 0.01
        self.W_dense = np.random.uniform(
            0, 1, size=(self.n_dim, self.vocab_size)) * 0.01

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
       
    def _ngs_train_step(self, center_word, context_words):
        
        center_word_idx = center_word
        black_list = np.concatenate((center_word_idx, context_words))
        context_size = context_words.shape[0]

        loss_pos = 0
        loss_neg = 0
        
        h = self.W_emb[center_word_idx]
            
        # Positive samples part
        c_pos_idx = context_words      
        c_pos = self.W_dense[:, c_pos_idx] 
        sigmoid_pos  = self._sigmoid(np.dot(h, c_pos))  
        grad_V_output_pos = (sigmoid_pos - 1) * h.T
        grad_V_input = np.sum((sigmoid_pos-1) * c_pos, axis=1)

        self.W_dense[:, c_pos_idx] = c_pos - self.learning_rate * grad_V_output_pos
        
        if self.compute_loss:
            loss_pos = np.sum(np.negative(np.log(sigmoid_pos)))

        # Negative samples part
        W_neg = np.random.choice(list(self.word_dist.keys()), replace=False, size=context_size+self.neg_samples+1, p=list(self.word_dist.values()))
        W_neg = list(set(W_neg) - set(black_list))[:self.neg_samples]
        
        c_neg = self.W_dense[:, W_neg]           
        sigmoid_neg = self._sigmoid(np.dot(h, c_neg))  
        grad_V_output_neg = sigmoid_neg * h.T
        grad_V_input += np.sum(sigmoid_neg * c_neg, axis=1)  
        
        self.W_dense[:, W_neg] = c_neg - self.learning_rate * grad_V_output_neg

        if self.compute_loss:
            loss_neg = np.sum(np.negative(np.log(self._sigmoid(np.dot(h, np.negative(c_neg))))))

        self.W_emb[center_word] = self.W_emb[center_word] - self.learning_rate * grad_V_input 

        context_loss = (loss_pos + loss_neg) / context_size
        return context_loss
    
    def train(self, text, epochs=None, compute_loss=False):

        if epochs is not None:
            self.epochs = epochs

        self.compute_loss = compute_loss

        if self.idx2word is None:  
            self._build_vocab(text)
            self._init_parameters()

            logging.info(f'Parameters (embedding) size in memory: {(self.W_emb.nbytes // 1024 // 1024) + (self.W_dense.nbytes // 1024 // 1024)} mb')

        curr_time = time()
        X, Y = self._preprocces_text(text)

        logging.info(f"Text preprocessing completed, X: {X.shape}, Y: {Y.shape} in {round(time()-curr_time, 4)} ms")
        logging.info(f"Samples size in memory: X {X.nbytes // 1024 // 1024} mb, Y {Y.nbytes // 1024 // 1024} mb")


        start_time = time()
        for epoch in range(self.epochs):
            loss = 0.0
            curr_time = time()
            for i in range(X.shape[0]):
                sample_loss = self._ngs_train_step(X[i].flatten(), Y[i][Y[i] > 0])
                if self.compute_loss:
                    loss += sample_loss
           
            logging.info(f"Epoch: {epoch}, loss: {loss / i if i > 0 else 1}, Time epoch: {time()-curr_time}")

        self.w2v = {self.idx2word[i]: vec for i, vec in enumerate(self.W_emb)}
        
        if self.save_output:
            with open('w2v.pickle', 'wb') as f:
                logging.info("Saving w2v.pickle...")
                pickle.dump(self.w2v, f)

        logging.info(f"Time: {time()-start_time}")

    def most_similar_words(self, word, threshold=None, tops=5):
        center_word_idx = self.word2idx[word]
        word_vec = self.w2v[word]        
        context_vectors = np.array([vec for _, vec in self.w2v.items()])        

        similarity_scores = context_vectors.dot(word_vec) / (np.linalg.norm(context_vectors, axis=1) * np.linalg.norm(word_vec))

        if threshold is not None:
            best_similarity_score = similarity_scores[similarity_scores > threshold]
            simmilars_idx = np.argwhere(similarity_scores > threshold).flatten()
            similars = sorted(zip(simmilars_idx, best_similarity_score), key=lambda x: x[1], reverse=True)            
        else:          
            similars = sorted(zip(range(similarity_scores.shape[0]), similarity_scores), key=lambda x: x[1], reverse=True)
        
        return [(self.idx2word[idx], score) for idx, score in similars[:tops+1] if idx != center_word_idx] 

def main():
    logging.root.setLevel('INFO')

    parser = ArgumentParser()
    parser.add_argument('--path', type=str, help="path to text file")
    parser.add_argument('--window_size', type=int, default=5, help="count of context words before and after center word")
    parser.add_argument('--learning_rate', type=float, default=0.05, help="optimizer hyperparameter")
    parser.add_argument('--n_dim', type=int, default=100, help="word vector size")
    parser.add_argument('--epochs', type=int, default=10,  help="train epochs count")
    parser.add_argument('--neg_samples', type=int, default=5, help="negative samples count")       
    parser.add_argument('--compute_loss', action='store_true', help="show loss value in logging output")
    parser.add_argument('--save_output', action='store_true', help="save pickle file with words embedding")

    args = parser.parse_args()
    
    with open(args.path, 'r', encoding='utf-8') as f:
        text = ' '.join(f.readlines())
        text = re.sub(r'([!\.,;:?="\(\)-])', r' \1 ', text)
        text = re.sub(r'[\t\n]', r' ', text)
    
    model_naive = NaiveWord2VecNGS(window_size=args.window_size, 
                                   learning_rate=args.learning_rate, 
                                   n_dim=args.n_dim, 
                                   epochs=args.epochs, 
                                   neg_samples=args.neg_samples, 
                                   save_output=args.save_output)
    
    model_naive.train(text, compute_loss=args.compute_loss)

if __name__ == '__main__':
    main()