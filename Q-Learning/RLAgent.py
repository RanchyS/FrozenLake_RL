import numpy as np
from matplotlib import pyplot as plt
import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
import pickle
import os

class QLearning():
    # Harita özelleştirme
    map_name = None#"8x8"
    size = 11
    p = 0.9
    seed = 3
    desc = generate_random_map(size= size,p= p,seed= seed)
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
    desc = None
        önceden tanımlanmış, map_size="4x4" ve map_size="8x8" haritaları
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

    q_table = None

    def train(self, episodes, is_slippery=False):
        global q_table
        # Parametreler
        epsilon = 1
        epsilon_to_zero = episodes * 9 / 10 # bu değerden sonra keşif(exploration) biter yalnızca sömürü(exploitaion) yapılır.
        learning_rate = 0.9
        discount_factor = 0.9
               
        env = gym.make('FrozenLake-v1', map_name = self.map_name, desc =self.desc, is_slippery=is_slippery, render_mode = None)
        
        q_table = np.zeros((env.observation_space.n, env.action_space.n))
        rewards_of_episodes = np.zeros(episodes)    

        for episode in range(episodes):

            state = env.reset()[0] # başlangıç durumunu döndürür.
            truncated= False
            terminated= False
            while(not(truncated or terminated)):

                if np.random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    # en büyük sayıyı bulma algoritması gereği bu ek koşul ifadesine ihtiyaç vardı(sayılar eşitse hep ilki döndürür!)
                    if q_table[state][0] == q_table[state][1] == q_table[state][2] == q_table[state][3]:
                        action = env.action_space.sample()
                    else:
                        action = q_table[state].argmax()

                new_state, reward, terminated, truncated, info = env.step(action)
                q_table[state][action] = q_table[state][action] + learning_rate * (reward + discount_factor * q_table[new_state].max() - q_table[state][action])
                state= new_state

                if reward == 1:
                    rewards_of_episodes[episode] = 1
                
            
            epsilon = epsilon - 1 / epsilon_to_zero

            if epsilon < 0.01: 
                learning_rate = 0.0001

        env.close()

        total_rewards = np.zeros(episodes)
        for t in range(episodes):
            total_rewards[t] = np.sum(rewards_of_episodes[max(0,t-99):t+1])

        plt.plot(total_rewards)
        plt.savefig("Reinforcement Learning\\rewards_and_episodes.png")
        
        file = open("Reinforcement Learning\\q_table.pkl","wb")
        pickle.dump(q_table, file)
        file.close()

    def test(self, n, is_slippery=False): # ortam üzerinde eğitilmiş ajanı n defa oynatır
        file = open("Reinforcement Learning\\q_table_11x11_slippery.pkl","rb")        
        q_table = pickle.load(file)
        file.close()

        env = gym.make('FrozenLake-v1', map_name = self.map_name, desc= self.desc, is_slippery= is_slippery, render_mode= "human")
        
        for i in range(n):
            terminated = False
            truncated = False
            
            state = env.reset()[0]
            steps = 0
            while not terminated and not truncated:
                action = q_table[state].argmax()
                new_state, reward, terminated, truncated, info = env.step(action)
                state = new_state
                
                if steps > 100: os._exit(0) # 8x8'lik haritada truncated flag'i 200. adımda kaldırılıyor. Diğerleri için default 100.
                steps += 1
        """ # son yüz episode'u detayli incelemek isteyenler bu kodun yorumunu açarak çalistirabilir.
            if truncated:
                print("maksimum adim siniri gecildi!", i+1, ". calistirma")
            else:
                if reward == 1:
                    print("hedefe ulasildi!", i+1, ". calistirma")
                else:
                    print("cukura dusuldu!", i+1, ". calistirma")
        """
        env.close()
        
agent = QLearning()
###agent.train(20000,is_slippery=True) #bir haritayı bir kere eğit, sonuçları gör ve beğen ardından bu satırı yorumlayarak çalıştır.
agent.test(2,is_slippery=True)
