import gymnasium as gym
import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import random
from collections import deque
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
import os

# hazır map kullanmak
map_name = None#"8x8"#None
# rastgele oluşturmak
size = 11
p = 0.9
seed = 3
desc = generate_random_map(size= size,p= p,seed= seed)
#desc = None
# kendimiz tanımlamak için
"""
desc = [ # istediğiniz haritayı oluşturabilirsiniz - rastgele ya da hazır haritaları kullanacaksanız burayı yorumlayınız!

        "SFFFFFFF",
        "FFFFFFFF",
        "FFFHFFFF",
        "FFFFFHFF",
        "FFFHFFFF",
        "FHGHFFHF",
        "FHFFHFHF",
        "FFFHFFFF",
        ]
"""

""" önceden tanımlanmış, map_size="4x4" ve map_size="8x8" haritaları
          "4x4":[
        "SFFF",
        "FHFH",
        "FFFH",
        "HFFG"
        ]
            "8x8": [
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFHFFFF",
        "FFFFFHFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFHFFFG",
        ]
"""

# 2️⃣ Sinir ağı tanımla
class DQN(nn.Module):
        def __init__(self, state_size, action_size):
            super().__init__()
            self.fc1 = nn.Linear(state_size, state_size) # harita boyutu kadar olan bir "girdi katmanı"
            self.fc2 = nn.Linear(state_size, state_size) # girdi katmanının nöron sayısı kadarlık bir "gizli katman"
            self.fc3 = nn.Linear(state_size, action_size) # eylem sayısı kadar nöronu bulunan "çıktı katmanı"

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            return self.fc3(x)  # Q-değerleri

def one_hot_encode(state, state_size):
        one_hot_vector = np.zeros(state_size)  # State_size uzunluğunda sıfırlarla dolu bir vektör oluştur
        one_hot_vector[state] = 1  # State'in olduğu yere 1 koy
        return torch.tensor(one_hot_vector, dtype=torch.float32)  # Tensör formatına çevir

def train(episodes,is_slippery=False):
    # 1️⃣ Ortamı oluştur
    env = gym.make('FrozenLake-v1', map_name = map_name, desc =desc, is_slippery=is_slippery, render_mode = None)

    # 3️⃣ Parametreler
    state_size = env.observation_space.n
    action_size = env.action_space.n
    learning_rate = 0.001
    gamma = 0.95
    epsilon = 1.0
    epsilon_to_zero = episodes * 9 / 10
    batch_size = 32
    memory = deque(maxlen=1000)

    # 4️⃣ Model ve hedef ağı oluştur
    model = DQN(state_size, action_size)
    target_model = DQN(state_size, action_size)
    target_model.load_state_dict(model.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 

    # 5️⃣ Deneyim Tekrarı (Experience Replay)
    def replay():
        if len(memory) < batch_size:
            return  # Hafızada yeterli veri yoksa çıkış yap

        minibatch = random.sample(memory, batch_size)  # Hafızadan rastgele örnekler seç

        # Tek döngü ile tüm verileri topluca ayır
        state_list, action_list, reward_list, next_state_list, done_list = [], [], [], [], []

        for exp in minibatch:
            state_list.append(exp[0])
            action_list.append(exp[1])
            reward_list.append(exp[2])
            next_state_list.append(exp[3])
            done_list.append(exp[4])

        # Tek seferde tensörlere dönüştür
        states = torch.stack(state_list)
        actions = torch.tensor(action_list, dtype=torch.int64)
        rewards = torch.tensor(reward_list, dtype=torch.float32)
        next_states = torch.stack(next_state_list)
        dones = torch.tensor(done_list, dtype=torch.float32)

        # Modelden mevcut Q-değerlerini al (seçilen aksiyona göre)
        current_q_values = model(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)

        # Hedef modelden gelecekteki en iyi Q-değerini al
        with torch.no_grad():  # Gradyan hesaplamalarını gereksiz yere tutmamak için
            next_q_values = target_model(next_states).max(dim=1)[0] #indeksler için [1]

        # Hedef Q-değerlerini hesapla (Bellman eşitliği)
        target_q_values = rewards + (gamma * next_q_values * (1 - dones)) # Gezinti sonlandıysa toplamanın sağ tarafı 0 olacaktır.

        # Kayıp fonksiyonunu hesapla (MSE Loss)
        loss = nn.functional.mse_loss(current_q_values, target_q_values)

        # Optimizasyonu uygula
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 6️⃣ Eğitim Döngüsü
    rewards = np.zeros(episodes)
    epsilons = np.zeros(episodes)
    flag = True
    for episode in range(episodes):
        state = env.reset()[0]
        #state = np.eye(state_size)[state]  # One-hot encoding, alternatif.
        state = one_hot_encode(state, state_size)
        done = False # not: gymnasium "terminated" şeklinde isimlendiriyor!
        truncated = False
        while not done and not truncated:
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()  # Rastgele hareket
            else:
                with torch.no_grad():
                  action = torch.argmax(model(state)).item()

            next_state, reward, done, truncated, _ = env.step(action)
            next_state = one_hot_encode(next_state, state_size)
            memory.append((state, action, reward, next_state, done))
            state = next_state

        replay()  # Deneyim tekrarını çalıştır

        if reward == 1:
          if flag: # en az bir kere ödül alındı mı? anlamsız güncellemeleri engellemek için flag değişkeni kullanıldı
            flag = False

          rewards[episode] = reward

        epsilons[episode] = epsilon

        epsilon = epsilon - 1 / epsilon_to_zero

        if episode % 5 == 0 and not flag: # eğitimi stabilleştirmek için 5 episode her iki ağı senkronize etmek için periyot olarak belirlendi.
          target_model.load_state_dict(model.state_dict())  # Hedef ağı güncelle

    env.close()

    total_rewards = np.zeros(episodes)
    for t in range(episodes):
        total_rewards[t] = np.sum(rewards[max(0,t-99):t+1]) # son 100 episode'da kazanılan ödül sayısı

    plt.plot(total_rewards)
    plt.grid(True)
    plt.savefig("Deep Reinforcement Learning\\rewards_and_episodes.png")
    plt.show()

    plt.plot(epsilons)
    plt.grid(True)
    plt.savefig("Deep Reinforcement Learning\\epsilons.png")
    plt.show()

    # policy network(model)'ü kaydetme
    torch.save(model.state_dict(), "Deep Reinforcement Learning\\frozen_lake_dql.pt")

def test(n,is_slippery=False): # ortam üzerinde eğitilmiş ajanı n defa oynatır

    env = gym.make('FrozenLake-v1', map_name = map_name, desc =desc, is_slippery=is_slippery, render_mode = "human")
    state_size = env.observation_space.n
    action_size = env.action_space.n

    model = DQN(state_size,action_size) 
    model.load_state_dict(torch.load("Deep Reinforcement Learning\\frozen_lake_dql_11x11_non_slippery_v2.pt"))
    model.eval()    # değerlendirme moduna geçiş

    for i in range(n):
         done = False
         truncated = False

         state = env.reset()[0]
         state = one_hot_encode(state,state_size)
         steps = 0
         while not done and not truncated:
            with torch.no_grad():
                action = torch.argmax(model(state)).item()
            state,reward,done,truncated,_ = env.step(action) # new_state'ten state'e atamak yerine doğrudan atadım.
            state = one_hot_encode(state,state_size)

            if steps > 200: os._exit(0) # buradaki değerle oynanabilir. Ek önlem almak adına oluşturuldu.
            steps += 1     

    env.close()

###train(10000,is_slippery=False) # bir haritayı bir kere eğit, sonuçları gör ve beğen ardından bu satırı yorumlayarak çalıştır.
test(2,is_slippery=False)
