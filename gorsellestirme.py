"""
# Eğitilmiş q_tablosunun çıktılarını görek için
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Eğitilmiş Q-tablosunu yükle
with open('Reinforcement Learning\\q_table_11x11_slippery.pkl', 'rb') as f:
    q_table = pickle.load(f)

# Harita boyutunu belirle
map_size = int(np.sqrt(q_table.shape[0]))

# Aksiyonlar ve ok sembolleri
action_arrows = {0: '←', 1: '↓', 2: '→', 3: '↑'}

# En iyi aksiyonları ve Q-değerlerini belirleme
best_actions = np.empty((map_size, map_size), dtype=str)
q_values = np.zeros((map_size, map_size))

for state in range(q_table.shape[0]):
    row, col = divmod(state, map_size)
    
    # Eğer tüm aksiyonların Q-değerleri aynıysa (örneğin hepsi 0), ok göstermeyelim!
    unique_values = np.unique(q_table[state])  # Tüm farklı Q-değerlerini al
    if len(unique_values) == 1:  # Eğer sadece tek bir farklı değer varsa (hepsi aynı), ok eklemiyoruz
        best_actions[row, col] = ""  # Boş bırak
    else:
        best_action = np.argmax(q_table[state])  # En büyük Q-değerine sahip aksiyon
        best_actions[row, col] = action_arrows[best_action]  # Ok sembolünü ekle

    q_values[row, col] = np.max(q_table[state])  # En büyük Q-değeri

# Görselleştirme
fig, ax = plt.subplots(figsize=(7, 7))
ax.imshow(q_values, cmap='Blues', vmin=np.min(q_values), vmax=np.max(q_values), zorder=1)

# **Izgarayı Düzenleme**
ax.set_xticks(np.arange(map_size + 1) - 0.5, minor=False)
ax.set_yticks(np.arange(map_size + 1) - 0.5, minor=False)
ax.grid(color="black", linestyle='-', linewidth=2, zorder=2)
ax.set_frame_on(False)

# **Çentikleri Kaldırma**
ax.tick_params(axis='both', which='both', length=0)
ax.set_xticklabels([])
ax.set_yticklabels([])

# **Okları Daha Net Hale Getirme**
for i in range(map_size):
    for j in range(map_size):
        if best_actions[i, j]:  # Eğer boş değilse, ok ekle
            ax.text(j, i, best_actions[i, j], ha='center', va='center', 
                    color='darkorange', fontsize=22, fontweight='bold', zorder=3)

# Isı haritası renk skalasını ekleyelim
fig.colorbar(ax.imshow(q_values, cmap='Blues', zorder=1), ax=ax)
ax.set_title("Optimal Policy: Oklar En İyi Hamleyi Gösterir")

plt.savefig("q_table_11x11_slippery.png", dpi=300, bbox_inches='tight') 
#plt.show()
"""
# Eğitilmiş DQN modelinin çıktısını göstermek için
import torch
import numpy as np
import matplotlib.pyplot as plt

# **Modeli Yükle**
class DQN(torch.nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_size, state_size)  
        self.fc2 = torch.nn.Linear(state_size, state_size)  
        self.fc3 = torch.nn.Linear(state_size, action_size)  

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)  # Q-değerleri

# **Harita Boyutunu Belirle**
map_size = 11  # Ortamın NxN boyutu

# **Eğitilmiş Modeli Yükle**
state_size = map_size * map_size  # Ortam boyutu (11x11)
action_size = 4  # Aksiyon sayısı
model = DQN(state_size, action_size)
model.load_state_dict(torch.load("Deep Reinforcement Learning\\frozen_lake_dql_11x11_slippery.pt"))
model.eval()  # Modeli değerlendirme moduna al

# **Aksiyonlar ve Ok Sembolleri**
action_arrows = {0: '←', 1: '↓', 2: '→', 3: '↑'}

# **En İyi Aksiyonları ve Q-değerlerini Belirleme**
best_actions = np.empty((map_size, map_size), dtype=str)
q_values = np.zeros((map_size, map_size))

# **One-Hot Encoding Fonksiyonu**
def one_hot_encode(state, state_size):
    one_hot_vector = np.zeros(state_size)
    one_hot_vector[state] = 1
    return torch.tensor(one_hot_vector, dtype=torch.float32)

# **Her Hücre İçin En İyi Aksiyonu Al**
for row in range(map_size):
    for col in range(map_size):
        state_index = row * map_size + col  # Satır ve sütundan tek boyutlu indeks oluştur
        state_tensor = one_hot_encode(state_index, state_size).unsqueeze(0)  # Model girişine uygun hale getir
        
        with torch.no_grad():
            q_values_tensor = model(state_tensor)  # Modelden Q-değerlerini al
        
        q_values[row, col] = q_values_tensor.max().item()  # En büyük Q-değerini al
        
        # Eğer tüm aksiyonların Q-değerleri aynıysa (örneğin hepsi 0), ok göstermeyelim!
        unique_values = np.unique(q_values_tensor.numpy())
        if len(unique_values) == 1:
            best_actions[row, col] = ""  # Ok eklenmesin
        else:
            best_action = torch.argmax(q_values_tensor).item()
            best_actions[row, col] = action_arrows[best_action]

# **Görselleştirme**
fig, ax = plt.subplots(figsize=(7, 7))
ax.imshow(q_values, cmap='Blues', vmin=np.min(q_values), vmax=np.max(q_values), zorder=1)

# **Izgarayı Düzenleme**
ax.set_xticks(np.arange(map_size + 1) - 0.5, minor=False)
ax.set_yticks(np.arange(map_size + 1) - 0.5, minor=False)
ax.grid(color="black", linestyle='-', linewidth=2, zorder=2)
ax.set_frame_on(False)

# **Çentikleri Kaldırma**
ax.tick_params(axis='both', which='both', length=0)
ax.set_xticklabels([])
ax.set_yticklabels([])

# **Okları Daha Net Hale Getirme**
for i in range(map_size):
    for j in range(map_size):
        if best_actions[i, j]:  # Eğer boş değilse, ok ekle
            ax.text(j, i, best_actions[i, j], ha='center', va='center', 
                    color='darkorange', fontsize=22, fontweight='bold', zorder=3)

# **Isı Haritası Renk Skalası ve Kaydetme**
fig.colorbar(ax.imshow(q_values, cmap='Blues', zorder=1), ax=ax)
ax.set_title("Optimal Policy: Oklar En İyi Hamleyi Gösterir")

plt.savefig("frozen_lake_dql_11x11_slippery.png", dpi=300, bbox_inches='tight')
#plt.show()
