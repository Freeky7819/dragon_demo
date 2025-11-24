import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
from PIL import Image
from torchvision import transforms
from sentence_transformers import SentenceTransformer

# ==========================================
# PART 1: TEXT ARCHITECTURE (The Brain)
# ==========================================

class ResonantPointer(nn.Module):
    """Basic unit for finding importance in text sequence."""
    def __init__(self, d_model: int, n_heads: int = 8, depth: int = 2, dropout: float = 0.05):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(d_model)
        self.final = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.norm(x)
        x = self.transformer(x)
        return self.final(x).squeeze(-1)

class MultiPhaseResonantPointer(nn.Module):
    """Advanced pointer with multiple phases and LSTM memory."""
    def __init__(self, d_model: int, n_phases: int = 2, total_depth: int = 4, dropout: float = 0.05):
        super().__init__()
        depth_per_phase = max(1, total_depth // n_phases)
        self.phases = nn.ModuleList([
            ResonantPointer(d_model=d_model, depth=depth_per_phase, dropout=dropout)
            for _ in range(n_phases)
        ])
        self.phase_projector = nn.Linear(d_model, d_model // 2)
        self.phase_memory = nn.LSTM(input_size=d_model // 2, hidden_size=d_model, num_layers=1, batch_first=True)
        self.confidence_gate = nn.Linear(d_model, 1)
        self.phase_weights = nn.Parameter(torch.ones(n_phases) / n_phases)
        self.residual_alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, hidden):
        B, T, D = hidden.shape
        accumulated_logits = torch.zeros(B, T, device=hidden.device)
        memory_state = None
        current_hidden = hidden
        weights = F.softplus(self.phase_weights)
        weights = weights / (weights.sum() + 1e-6)
        
        for i, pointer in enumerate(self.phases):
            phase_scores = pointer(current_hidden)
            gate_raw = self.confidence_gate(current_hidden)
            confidence = torch.sigmoid(gate_raw.squeeze(-1) * 8.0)
            accumulated_logits += phase_scores * confidence * weights[i]
            
            summary = self.phase_projector(current_hidden.mean(dim=1, keepdim=True))
            lstm_out, memory_state = self.phase_memory(summary, memory_state if i > 0 else None)
            current_hidden = hidden + self.residual_alpha * lstm_out.expand(-1, T, -1)
            
        return accumulated_logits

class DragonArchitecture(nn.Module):
    """Main architecture for text compression."""
    def __init__(self, d_model=384, max_seq_len=128):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # 1. Pointer System
        self.pointer = MultiPhaseResonantPointer(d_model=d_model, n_phases=2, total_depth=4)
        
        # 2. Context Mixer
        self.neighbor_mixer = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model//32),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=2, dilation=2, groups=d_model//32),
        )
        
        # 3. Reconstruction (REQUIRED FOR LOADING WEIGHTS)
        # We don't use this for search, but it must be here because it's in the .pth file
        self.residual = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model),
        )
        self.pos_bias = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        
        # 4. Parameters
        self.harmonic_w = nn.Parameter(torch.tensor(0.7))
        self.gamma = 0.0025
        self.ln = nn.LayerNorm(d_model)

    def harmonic_injection(self, x):
        T = x.shape[1]
        pos = torch.arange(T, device=x.device).float()
        sig = torch.exp(-self.gamma * pos) * torch.sin(6.0 * pos + math.pi/3)
        return x + self.harmonic_w * sig.unsqueeze(0).unsqueeze(-1)

    def compress(self, x, ratio: int = 16):
        B, T, D = x.shape
        k = max(1, T // ratio)
        h = self.harmonic_injection(x)
        logits = self.pointer(h)
        vals, top_indices = logits.topk(k, dim=1)
        m = self.neighbor_mixer(h.transpose(1,2)).transpose(1,2)
        compressed = m.gather(1, top_indices.unsqueeze(-1).expand(-1,-1, D))
        gate = torch.sigmoid(vals).unsqueeze(-1)
        compressed = self.ln(compressed * gate)
        return compressed, top_indices.float() / self.max_seq_len

    def compress_adaptive(self, x, sensitivity=0.2, min_vectors=2):
        B, T, D = x.shape
        padding_mask = x.abs().sum(dim=-1) > 1e-9
        h = self.harmonic_injection(x)
        logits = self.pointer(h).squeeze(-1)
        batch_results = []
        
        for i in range(B):
            # Get valid positions first
            valid_pos = padding_mask[i].nonzero(as_tuple=False).squeeze(-1)
            if len(valid_pos) == 0:
                indices = torch.tensor([0], device=x.device)
            else:
                valid_logits = logits[i][valid_pos]
                mean, std = valid_logits.mean(), valid_logits.std()
                threshold = mean + (sensitivity * std)
                mask = (logits[i] > threshold) & padding_mask[i]
                indices = mask.nonzero(as_tuple=False).squeeze(-1)
                
                # Fallback: if not enough indices, use top-k only over valid tokens
                if len(indices) < min_vectors:
                    k = min(min_vectors, valid_logits.numel())
                    _, top_rel = valid_logits.topk(k)
                    indices = valid_pos[top_rel]
            
            indices, _ = torch.sort(indices)
            m = self.neighbor_mixer(h[i].unsqueeze(0).transpose(1,2)).transpose(1,2)
            selected = m[0, indices, :]
            gate = torch.sigmoid(logits[i, indices]).unsqueeze(-1)
            selected = self.ln(selected * gate)
            batch_results.append({
                "compressed_vectors": selected, 
                "positions": indices.float() / self.max_seq_len, 
                "count": len(indices)
            })
        return batch_results

# ==========================================
# PART 2: VISION ARCHITECTURE (The Eyes)
# ==========================================

class HexagonalHarmonicInjection(nn.Module):
    """Injects 3-axis hexagonal resonance into image patches."""
    def __init__(self, d_model, grid_size=14, omega=6.0):
        super().__init__()
        self.d_model, self.grid_size, self.omega = d_model, grid_size, omega
        self.angles = [0, 2 * math.pi / 3, 4 * math.pi / 3]

    def _generate_hex_grid(self, batch_size, device):
        gs = self.grid_size
        coords = torch.linspace(-1, 1, gs, device=device)
        y, x = torch.meshgrid(coords, coords, indexing='ij')
        x, y = x.flatten(), y.flatten()
        pe = torch.zeros(gs * gs, self.d_model, device=device)
        div_term = torch.exp(torch.arange(0, self.d_model, 2, device=device).float() * (-math.log(10000.0) / self.d_model))
        total_wave = torch.zeros(gs * gs, self.d_model // 2, device=device)

        for theta in self.angles:
            axis_proj = x * math.cos(theta) + y * math.sin(theta)
            phase = axis_proj.unsqueeze(1) * self.omega * div_term.unsqueeze(0)
            total_wave += phase

        pe[:, 0::2] = torch.sin(total_wave)
        pe[:, 1::2] = torch.cos(total_wave)
        return pe.unsqueeze(0).expand(batch_size, -1, -1)

    def forward(self, x):
        return x + self._generate_hex_grid(x.shape[0], x.device)

class DragonVision(nn.Module):
    """Vision Transformer with Hexagonal Logic."""
    def __init__(self, d_model=384, patch_size=16, image_size=224):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, d_model, kernel_size=patch_size, stride=patch_size)
        self.hex_pos = HexagonalHarmonicInjection(d_model, grid_size=image_size // patch_size)
        self.ln_pre = nn.LayerNorm(d_model)
        self.compressor = DragonArchitecture(d_model=d_model, max_seq_len=(image_size//patch_size)**2)

    def forward(self, x, ratio=16):
        x = self.patch_embed(x).flatten(2).transpose(1, 2) # [B, Patches, D]
        x = self.ln_pre(self.hex_pos(x))
        return self.compressor.compress(x, ratio=ratio)

# ==========================================
# PART 3: UNIFIED INTERFACE
# ==========================================

def _safe_torch_load(path, device):
    """Safely load PyTorch checkpoint with weights_only=True if supported."""
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        # Fallback for older PyTorch versions that don't support weights_only
        return torch.load(path, map_location=device)

class Dragon:
    def __init__(self, model_dir="models", device=None, vision_mode=False):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.vision_mode = vision_mode
        
        if not vision_mode:
            self.nlp = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
            self.model = DragonArchitecture(d_model=384).to(self.device)
            path = os.path.join(model_dir, 'dragon_pro_1_16.pth')
            if os.path.exists(path):
                print(f"📂 Loading TEXT Model from {path}")
                try:
                    state = _safe_torch_load(path, self.device)
                    self.model.load_state_dict(state, strict=True)
                except Exception as e:
                    print(f"⚠️ Error loading model weights: {e}")
                    print("⚠️ Using random initialization.")
            else:
                print("⚠️ WARNING: Text model weights not found. Using random initialization.")
            self.model.eval()
        else:
            self.vision_model = DragonVision(d_model=384).to(self.device)
            path = os.path.join(model_dir, 'dragon_vision_v1.pth')
            if os.path.exists(path):
                print(f"👁️ Loading VISION Model from {path}")
                try:
                    state = _safe_torch_load(path, self.device)
                    # Allow missing keys for vision if exact match isn't critical
                    self.vision_model.load_state_dict(state, strict=False)
                except Exception as e:
                    print(f"⚠️ Error loading vision model weights: {e}")
                    print("⚠️ Using random initialization.")
            else:
                 print("⚠️ WARNING: Vision model weights not found. Using random initialization.")
            self.vision_model.eval()
            self.img_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def compress(self, text_or_list, adaptive=True, sensitivity=1.0, min_vectors=2):
        if self.vision_mode: raise ValueError("In Vision Mode")
        
        token_emb = self.nlp.encode(text_or_list, output_value='token_embeddings', convert_to_tensor=True)
        batch_tensors = token_emb if isinstance(token_emb, list) else [token_emb]
        
        target_len = 128
        padded = torch.zeros(len(batch_tensors), target_len, 384).to(self.device)
        for i, t in enumerate(batch_tensors):
            l = min(t.shape[0], target_len)
            padded[i, :l, :] = t[:l, :]
            
        with torch.no_grad():
            return self.model.compress_adaptive(
                padded,
                sensitivity=sensitivity,
                min_vectors=min_vectors
            )

    def compress_image(self, image_path_or_pil, ratio=16):
        if not self.vision_mode: raise ValueError("Not in Vision Mode")
        
        img = Image.open(image_path_or_pil).convert('RGB') if isinstance(image_path_or_pil, str) else image_path_or_pil
        img_tensor = self.img_transform(img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            compressed, positions = self.vision_model(img_tensor, ratio=ratio)
            
        return {"compressed_vectors": compressed.cpu(), "positions": positions.cpu(), "ratio": ratio}
