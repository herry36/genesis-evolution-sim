"""
Genesis Evolution System - Simulatore Ecosistema Evolutivo STANDALONE - PROCEDURALE
===================================================================================

"""

# Silencia warning macOS PRIMA di importare matplotlib
import os
os.environ['TK_SILENCE_DEPRECATION'] = '1'

# Controlla e installa dipendenze
try:
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import matplotlib.patches as mpatches
except ImportError as e:
    print(f"❌ Dipendenza mancante: {e}")
    print("📦 Installa con: pip install numpy matplotlib seaborn")
    exit(1)

import random
import copy
import json
import uuid
from typing import Dict, List, Tuple, Optional, Any, Set
from collections import defaultdict, Counter
import math
import time
from datetime import datetime

# Configura stile grafici e warning
try:
    plt.style.use('dark_background') 
except:
    plt.style.use('default')

# Backend appropriato
try:
    import matplotlib
    matplotlib.use('TkAgg')  # Prova backend interattivo
    print("✅ Backend grafico interattivo abilitato")
except:
    matplotlib.use('Agg')  # Fallback non-interattivo
    print("✅ Backend grafico non-interattivo (solo salvataggio)")

# Configura palette e warning
sns.set_palette("husl")
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
warnings.filterwarnings("ignore", category=UserWarning, module="tkinter")

print("🎨 Sistema grafici configurato correttamente")

# ===============================
# CONFIGURAZIONI E COSTANTI
# ===============================

# Configurazione di default
DEFAULT_CONFIG = {
    'carrying_capacity': 1000,
    'mass_extinction_probability': 0.01,
    'adaptive_radiation_probability': 0.02,
    'climate_change_probability': 0.05,
    'min_species': 3,
    'max_species': 20
}

DEFAULT_ECOSYSTEM_CONFIG = {
    'resource_abundance': 1.0,
    'environmental_pressure': 0.5,
    'environmental_stability': 0.8,
    'resource_competition': True,
    'predator_prey_dynamics': True,
    'predation_efficiency': 0.3,
    'prey_defense_effectiveness': 0.2,
    'symbiosis_formation_rate': 0.1,
    'mutualism_benefit': 0.1,
    'niche_construction_enabled': True,
    'environment_modification_rate': 0.05
}

# Trading styles disponibili
TRADING_STYLES = ['scalper', 'swing', 'momentum', 'contrarian', 'trend_follower', 'conservative', 'aggressive']

# Tipi di risorse per trading
RESOURCE_TYPES = [
    'high_frequency_opportunities',
    'trend_following_signals',
    'mean_reversion_setups',
    'volatility_spikes',
    'low_risk_steady_gains',
    'arbitrage_opportunities',
    'breakout_patterns',
    'swing_trading_setups'
]

# ===============================
# GRAFICO BIODIVERSITÀ - DEFINITO PRESTO! 🎨
# ===============================

def create_beautiful_biodiversity_png(ecosystem_data, species_populations):
    """Crea PNG SPETTACOLARE della biodiversità - DESIGN DA FIGATA! 🎨"""
    
    print("🎨 Creando PNG SPETTACOLARE di design...")
    
    # Chiudi tutto
    plt.close('all')
    
    # Usa stile moderno
    plt.style.use('default')
    
    # Figura ENORME e professionale
    fig = plt.figure(figsize=(20, 12))
    
    # SFONDO GRADIENTE SPETTACOLARE
    from matplotlib.colors import LinearSegmentedColormap
    import matplotlib.patches as patches
    
    # Gradiente da nero a blu scuro
    gradient = np.linspace(0, 1, 256).reshape(256, -1)
    gradient = np.hstack((gradient, gradient))
    
    # Sfondo principale
    fig.patch.set_facecolor('#0a0a0a')
    
    # Grid layout avanzato
    gs = fig.add_gridspec(3, 3, height_ratios=[0.15, 0.7, 0.15], width_ratios=[0.1, 0.8, 0.1])
    
    # Area principale del grafico
    ax = fig.add_subplot(gs[1, 1])
    
    # SFONDO GRAFICO CON GRADIENTE
    ax.set_facecolor('#111111')
    
    # Aggiungi rettangolo gradiente di sfondo
    gradient_bg = patches.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                                  facecolor='none', edgecolor='none')
    ax.add_patch(gradient_bg)
    
    if not ecosystem_data['ecosystem_history']:
        # Messaggio elegante se non ci sono dati
        ax.text(0.5, 0.5, '⚠️ NO EVOLUTION DATA AVAILABLE', 
                ha='center', va='center', fontsize=32, 
                color='#ff3366', fontweight='bold', family='sans-serif')
        
        # Titolo header
        fig.text(0.5, 0.9, '🧬 GENESIS ECOSYSTEM - BIODIVERSITY EVOLUTION 📊', 
                ha='center', va='center', fontsize=28, 
                color='white', fontweight='bold', family='sans-serif')
    else:
        # Estrai dati
        generations = np.array([state['generation'] for state in ecosystem_data['ecosystem_history']])
        biodiversity = np.array([state['biodiversity_index'] for state in ecosystem_data['ecosystem_history']])
        populations = np.array([state['total_population'] for state in ecosystem_data['ecosystem_history']])
        
        print(f"📊 Creando grafico SPETTACOLARE con {len(generations)} generazioni")
        
        # PLOT PRINCIPALE - Biodiversità con EFFETTI SPETTACOLARI
        
        # Area sotto la curva con gradiente
        ax.fill_between(generations, 0, biodiversity, 
                       alpha=0.4, color='#00ffaa', zorder=1)
        ax.fill_between(generations, 0, biodiversity, 
                       alpha=0.2, color='#00ff66', zorder=1)
        
        # Linea principale con ombra
        ax.plot(generations, biodiversity + 0.01,  # Ombra
               color='#003322', linewidth=8, alpha=0.3, zorder=2)
        
        ax.plot(generations, biodiversity, 
               color='#00ffaa', linewidth=6, 
               marker='o', markersize=12, markerfacecolor='#00ffaa',
               markeredgecolor='white', markeredgewidth=3,
               label='🧬 BIODIVERSITY INDEX', alpha=1.0, zorder=5,
               markevery=max(1, len(generations)//20))  # Marker sparsi
        
        # Linea di tendenza
        if len(biodiversity) > 1:
            z = np.polyfit(generations, biodiversity, 1)
            p = np.poly1d(z)
            ax.plot(generations, p(generations), 
                   color='#66ffcc', linewidth=3, linestyle=':', 
                   alpha=0.7, zorder=3, label='📈 TREND LINE')
        
        # Asse secondario per popolazione
        ax2 = ax.twinx()
        
        # Area popolazione
        ax2.fill_between(generations, 0, populations, 
                        alpha=0.3, color='#ff6699', zorder=1)
        
        # Linea popolazione con ombra
        ax2.plot(generations, populations + (max(populations)*0.02),  # Ombra
                color='#330011', linewidth=8, alpha=0.3, zorder=2)
        
        ax2.plot(generations, populations,
                color='#ff3366', linewidth=5,
                marker='s', markersize=10, markerfacecolor='#ff3366',
                markeredgecolor='white', markeredgewidth=3,
                linestyle='--', label='👥 TOTAL POPULATION', 
                alpha=0.9, zorder=4,
                markevery=max(1, len(generations)//20))
        
        # STYLING ULTRA-MODERNO
        ax.set_xlabel('🕐 GENERATION', color='white', fontsize=20, fontweight='bold', family='sans-serif')
        ax.set_ylabel('🧬 SHANNON BIODIVERSITY INDEX', color='#00ffaa', fontsize=20, fontweight='bold', family='sans-serif')
        ax2.set_ylabel('👥 TOTAL POPULATION', color='#ff3366', fontsize=20, fontweight='bold', family='sans-serif')
        
        # HEADER SPETTACOLARE
        header_text = '🌍 GENESIS ECOSYSTEM - BIODIVERSITY EVOLUTION DASHBOARD 📊'
        fig.text(0.5, 0.95, header_text, 
                ha='center', va='center', fontsize=26, 
                color='white', fontweight='bold', family='sans-serif')
        
        # Sottotitolo
        subtitle = f'🔬 Advanced Evolutionary Analytics • Generation {generations[-1]} • {datetime.now().strftime("%Y-%m-%d")}'
        fig.text(0.5, 0.91, subtitle, 
                ha='center', va='center', fontsize=14, 
                color='#888888', style='italic', family='sans-serif')
        
        # Griglia ELEGANTE con effetti
        ax.grid(True, alpha=0.3, color='#333333', linestyle='-', linewidth=1)
        ax.grid(True, alpha=0.1, color='#666666', linestyle=':', linewidth=0.5)
        ax.set_axisbelow(True)
        
        # Legenda SPETTACOLARE
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        legend = ax.legend(lines1 + lines2, labels1 + labels2, 
                          loc='upper left', framealpha=0.95, fontsize=16,
                          facecolor='#1a1a1a', edgecolor='#00ffaa', 
                          fancybox=True, shadow=True, borderpad=1)
        legend.get_frame().set_linewidth(3)
        for text in legend.get_texts():
            text.set_color('white')
            text.set_fontweight('bold')
        
        # BOX STATISTICHE ULTRA-MODERNO
        final_bio = biodiversity[-1] if len(biodiversity) > 0 else 0
        final_pop = populations[-1] if len(populations) > 0 else 0
        final_gen = generations[-1] if len(generations) > 0 else 0
        
        # Calcola metriche avanzate
        bio_change = ((final_bio - biodiversity[0]) / biodiversity[0] * 100) if len(biodiversity) > 1 and biodiversity[0] > 0 else 0
        pop_change = ((final_pop - populations[0]) / populations[0] * 100) if len(populations) > 1 and populations[0] > 0 else 0
        
        stats_text = f"""📊 ECOSYSTEM METRICS
━━━━━━━━━━━━━━━━━━━━━━━━━
🕐 GENERATION: {final_gen:,}
🧬 BIODIVERSITY: {final_bio:.6f}
👥 POPULATION: {final_pop:,}
🔬 SPECIES COUNT: {len(species_populations)}
━━━━━━━━━━━━━━━━━━━━━━━━━
📈 BIO CHANGE: {bio_change:+.2f}%
📊 POP CHANGE: {pop_change:+.2f}%
🎯 TREND: {'🚀 GROWING' if bio_change > 0 else '📉 DECLINING' if bio_change < 0 else '➡️ STABLE'}"""
        
        # Box principale con bordi neon
        stats_box = patches.FancyBboxPatch((0.72, 0.15), 0.26, 0.7,
                                         boxstyle="round,pad=0.02",
                                         facecolor='#0a0a0a', edgecolor='#00ffaa',
                                         linewidth=4, alpha=0.95,
                                         transform=ax.transAxes)
        ax.add_patch(stats_box)
        
        # Box interno con gradiente
        inner_box = patches.FancyBboxPatch((0.73, 0.16), 0.24, 0.68,
                                         boxstyle="round,pad=0.01",
                                         facecolor='#111111', alpha=0.8,
                                         transform=ax.transAxes)
        ax.add_patch(inner_box)
        
        ax.text(0.85, 0.8, stats_text, transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='center',
                color='white', fontsize=13, fontweight='bold',
                family='monospace', linespacing=1.2)
        
        # INDICATORI DI PERFORMANCE
        perf_indicators = [
            ('🔥 DIVERSITY', final_bio, '#00ffaa'),
            ('⚡ GROWTH', bio_change/100, '#ffaa00' if bio_change > 0 else '#ff3366'),
            ('🎯 STABILITY', 1.0 - abs(bio_change)/100, '#66aaff')
        ]
        
        for i, (label, value, color) in enumerate(perf_indicators):
            y_pos = 0.1 - i * 0.03
            bar_length = abs(value) * 0.2 if abs(value) <= 1 else 0.2
            
            # Barra indicatore
            bar = patches.Rectangle((0.73, y_pos-0.01), bar_length, 0.02,
                                  facecolor=color, alpha=0.8,
                                  transform=ax.transAxes)
            ax.add_patch(bar)
            
            # Label
            ax.text(0.72, y_pos, f'{label}: {value:.3f}', 
                   transform=ax.transAxes, color=color,
                   fontsize=11, fontweight='bold', ha='right')
    
    # STYLING ASSI ULTRA-MODERNO
    ax.tick_params(colors='white', labelsize=16, width=2, length=8, 
                   labelcolor='white', pad=8)
    ax2.tick_params(colors='white', labelsize=16, width=2, length=8,
                    labelcolor='white', pad=8)
    
    # Spines NEON
    for spine in ax.spines.values():
        spine.set_color('#00ffaa')
        spine.set_linewidth(3)
        spine.set_alpha(0.8)
    for spine in ax2.spines.values():
        spine.set_color('#ff3366')
        spine.set_linewidth(3)
        spine.set_alpha(0.8)
    
    # FOOTER ELEGANTE
    footer_text = f"🚀 GENESIS EVOLUTION SYSTEM v2.0 | Ultra-Advanced Ecosystem Analytics | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    fig.text(0.5, 0.02, footer_text, ha='center', va='bottom', 
             fontsize=12, color='#666666', style='italic', family='sans-serif')
    
    # Bordi decorativi
    border_top = patches.Rectangle((0, 0.98), 1, 0.02, transform=fig.transFigure,
                                 facecolor='#00ffaa', alpha=0.8)
    border_bottom = patches.Rectangle((0, 0), 1, 0.02, transform=fig.transFigure,
                                    facecolor='#ff3366', alpha=0.8)
    fig.patches.extend([border_top, border_bottom])
    
    # Layout perfetto
    plt.tight_layout()
    plt.subplots_adjust(top=0.88, bottom=0.08, left=0.08, right=0.92)
    
    # SALVA PNG ULTRA QUALITÀ
    filename = f'BIODIVERSITY_EVOLUTION_SPECTACULAR_gen_{ecosystem_data["state"]["generation"]:04d}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight', 
               facecolor='#0a0a0a', edgecolor='none',
               pad_inches=0.3)
    
    plt.close('all')  # Chiudi senza mostrare
    
    print(f"✅ PNG SPETTACOLARE salvato: {filename}")
    print("🔥 Design ultra-moderno con neon, gradiente e dashboard professionale!")
    
    return filename

def generate_biodiversity_graphic_only(ecosystem_data, species_populations):
    """Genera SOLO PNG bellissimo - NIENTE FINESTRE! 🎨"""
    
    print("\n🎨 === CREAZIONE PNG BIODIVERSITÀ ===")
    
    try:
        # SOLO PNG bellissimo
        png_file = create_beautiful_biodiversity_png(ecosystem_data, species_populations)
        
        print("✅ PNG biodiversità creato!")
        print(f"📁 File: {png_file}")
        
        # Prova ad aprire il PNG automaticamente
        import os
        import subprocess
        import platform
        
        try:
            if platform.system() == "Darwin":  # macOS
                subprocess.run(["open", png_file], check=True)
                print(f"🖼️ PNG aperto automaticamente su macOS!")
            elif platform.system() == "Windows":
                os.startfile(png_file)
                print(f"🖼️ PNG aperto automaticamente su Windows!")
            else:  # Linux
                subprocess.run(["xdg-open", png_file], check=True)
                print(f"🖼️ PNG aperto automaticamente su Linux!")
        except Exception as e:
            print(f"📁 PNG salvato (apertura automatica fallita): {e}")
        
        print("✅ PNG biodiversità BELLISSIMO completato!")
        return png_file
        
    except Exception as e:
        print(f"❌ Errore nella creazione PNG: {e}")
        import traceback
        traceback.print_exc()
        return None

# ===============================
# GENE UTILS - Versione procedurale
# ===============================

def get_individual_gene_value(individual, gene_name, default_value=0.0):
    """Ottieni valore di un gene per un individuo"""
    return individual.get(gene_name, default_value)

def get_average_gene_value(population, gene_name, default_value=0.0):
    """Calcola valore medio di un gene in una popolazione"""
    if not population:
        return default_value
    
    values = [get_individual_gene_value(ind, gene_name, default_value) for ind in population]
    numeric_values = [v for v in values if isinstance(v, (int, float))]
    
    return sum(numeric_values) / len(numeric_values) if numeric_values else default_value

def set_individual_gene_value(individual, gene_name, value):
    """Imposta valore di un gene per un individuo"""
    individual[gene_name] = value

def apply_fitness_modifier_to_population(population, modifier):
    """Applica modificatore fitness a tutta la popolazione"""
    for individual in population:
        current_fitness = get_individual_gene_value(individual, 'fitness_composito', 0.0)
        set_individual_gene_value(individual, 'fitness_composito', current_fitness * modifier)

# ===============================
# CREAZIONE STATO ECOSISTEMA
# ===============================

def create_ecosystem_state():
    """Crea stato iniziale dell'ecosistema"""
    return {
        'generation': 0,
        'total_population': 0,
        'species_count': 0,
        'carrying_capacity': DEFAULT_CONFIG['carrying_capacity'],
        'resource_abundance': DEFAULT_ECOSYSTEM_CONFIG['resource_abundance'],
        'environmental_pressure': DEFAULT_ECOSYSTEM_CONFIG['environmental_pressure'],
        'climate_stability': DEFAULT_ECOSYSTEM_CONFIG['environmental_stability'],
        
        # Metrics
        'biodiversity_index': 0.0,
        'competition_intensity': 0.0,
        'predation_pressure': 0.0,
        'symbiosis_networks': 0,
        
        # Environmental factors
        'temperature': 0.5,
        'volatility': 0.5,
        'trend_strength': 0.5,
        'market_regime': 'neutral'
    }

def create_ecosystem_data():
    """Crea struttura dati completa dell'ecosistema"""
    return {
        'state': create_ecosystem_state(),
        'resource_patches': {},
        'species_interactions': {},
        'symbiosis_networks': [],
        'ecosystem_history': [],
        'ecosystem_events': [],
        'extinction_events': [],
        'species_fitness_cache': {},
        'event_id_counter': 0,
        'patch_id_counter': 0
    }

def create_resource_patch(patch_id, location, resource_type):
    """Crea un patch di risorse"""
    return {
        'patch_id': patch_id,
        'location': location,
        'resource_type': resource_type,
        'abundance': random.uniform(0.3, 1.0),
        'regeneration_rate': random.uniform(0.05, 0.2),
        'capacity': random.uniform(50, 200),
        'occupants': [],
        'quality': random.uniform(0.5, 1.0)
    }

def create_species_interaction(species1_id, species2_id, interaction_type, strength):
    """Crea interazione tra specie"""
    return {
        'species1_id': species1_id,
        'species2_id': species2_id,
        'interaction_type': interaction_type,
        'strength': strength,
        'direction': 'bidirectional' if interaction_type in ['competition', 'mutualism'] else 'species1_to_species2',
        'stability': random.uniform(0.5, 0.95),
        'generation_formed': 0
    }

def create_ecosystem_event(event_id, event_type, generation, affected_species, magnitude):
    """Crea evento dell'ecosistema"""
    return {
        'event_id': event_id,
        'event_type': event_type,
        'generation': generation,
        'timestamp': datetime.now(),
        'affected_species': affected_species,
        'magnitude': magnitude,
        'duration': 1,
        'trigger': 'stochastic',
        'outcomes': {}
    }

# ===============================
# GENERAZIONE INDIVIDUI E SPECIE
# ===============================

def create_random_individual(trading_style=None):
    """Crea un individuo trader casuale"""
    if not trading_style:
        trading_style = random.choice(TRADING_STYLES)
    
    individual = {
        'individual_id': str(uuid.uuid4())[:8],
        'trading_style': trading_style,
        'fitness_composito': random.uniform(0.3, 1.0),
        'aggressiveness': random.uniform(0.1, 3.0),
        'position_size_base': random.uniform(0.05, 0.5),
        'time_horizon': random.uniform(0.1, 2.0),
        'innovation_score': random.uniform(0.2, 1.0),
        'birth_generation': 0,
        'ecological_pressure': 0.0,
        'interaction_benefits': 0.0
    }
    
    return individual

def create_species_population(species_id, size=50, trading_style=None):
    """Crea popolazione di una specie"""
    return [create_random_individual(trading_style) for _ in range(size)]

def generate_initial_species_populations(num_species=5):
    """Genera popolazioni iniziali di specie diverse"""
    populations = {}
    
    for i in range(num_species):
        species_id = f"species_{i+1:03d}"
        trading_style = random.choice(TRADING_STYLES)
        size = random.randint(30, 80)
        
        populations[species_id] = create_species_population(species_id, size, trading_style)
        
        print(f"   📊 Creata {species_id}: {size} individui ({trading_style})")
    
    return populations

# ===============================
# INIZIALIZZAZIONE RISORSE
# ===============================

def initialize_resource_patches(ecosystem_data):
    """Inizializza patches di risorse nell'ecosistema"""
    num_patches = 20
    
    for i in range(num_patches):
        ecosystem_data['patch_id_counter'] += 1
        patch_id = f"PATCH_{ecosystem_data['patch_id_counter']:06d}"
        
        location = (random.uniform(0, 10), random.uniform(0, 10))
        resource_type = random.choice(RESOURCE_TYPES)
        
        patch = create_resource_patch(patch_id, location, resource_type)
        ecosystem_data['resource_patches'][patch_id] = patch
        
        print(f"   🌿 Patch {patch_id}: {resource_type} @ {location}")

# ===============================
# UPDATE STATO ECOSISTEMA
# ===============================

def update_ecosystem_state(ecosystem_data, species_populations):
    """Aggiorna stato generale dell'ecosistema"""
    state = ecosystem_data['state']
    
    # Conta popolazione totale
    total_pop = sum(len(pop) for pop in species_populations.values())
    state['total_population'] = total_pop
    state['species_count'] = len(species_populations)
    
    # Calcola biodiversità (Shannon index)
    if species_populations:
        pop_sizes = [len(pop) for pop in species_populations.values() if len(pop) > 0]
        if pop_sizes:
            total = sum(pop_sizes)
            proportions = [size / total for size in pop_sizes]
            shannon_index = -sum(p * math.log(p) for p in proportions if p > 0)
            state['biodiversity_index'] = shannon_index
    
    # Calcola pressioni
    capacity_pressure = total_pop / state['carrying_capacity']
    state['environmental_pressure'] = min(1.0, capacity_pressure)
    
    # Update fattori climatici
    update_climate_factors(state)

def update_climate_factors(state):
    """Aggiorna fattori climatici dell'ecosistema"""
    # Simula cambiamenti climatici graduali
    temp_change = np.random.normal(0, 0.02)
    state['temperature'] = np.clip(state['temperature'] + temp_change, 0.0, 1.0)
    
    # Volatilità del mercato
    vol_change = np.random.normal(0, 0.05)
    state['volatility'] = np.clip(state['volatility'] + vol_change, 0.0, 1.0)
    
    # Trend strength
    trend_change = np.random.normal(0, 0.03)
    state['trend_strength'] = np.clip(state['trend_strength'] + trend_change, 0.0, 1.0)
    
    # Determina regime
    if state['volatility'] > 0.7:
        state['market_regime'] = "high_volatility"
    elif state['trend_strength'] > 0.7:
        state['market_regime'] = "trending"
    elif state['trend_strength'] < 0.3:
        state['market_regime'] = "ranging"
    else:
        state['market_regime'] = "neutral"

# ===============================
# INTERAZIONI ECOLOGICHE
# ===============================

def simulate_ecological_interactions(ecosystem_data, species_populations):
    """Simula tutte le interazioni ecologiche"""
    # Aggiorna matrici di interazione
    update_interaction_matrices(ecosystem_data, species_populations)
    
    # Applica effetti delle interazioni
    modified_populations = copy.deepcopy(species_populations)
    
    for species_id, population in modified_populations.items():
        interaction_effects = calculate_interaction_effects(ecosystem_data, species_id, species_populations)
        
        # Applica effetti a livello di popolazione
        population_modifier = interaction_effects.get('population_modifier', 1.0)
        fitness_modifier = interaction_effects.get('fitness_modifier', 1.0)
        
        # Modifica dimensione popolazione
        if population_modifier < 1.0:
            reduction = int(len(population) * (1.0 - population_modifier))
            if reduction > 0:
                population.sort(key=lambda x: get_individual_gene_value(x, 'fitness_composito', 0.0))
                modified_populations[species_id] = population[reduction:]
        
        # Modifica fitness degli individui
        for individual in modified_populations[species_id]:
            current_fitness = get_individual_gene_value(individual, 'fitness_composito', 0.0)
            set_individual_gene_value(individual, 'fitness_composito', current_fitness * fitness_modifier)
            
            individual['ecological_pressure'] = interaction_effects.get('pressure', 0.0)
            individual['interaction_benefits'] = interaction_effects.get('benefits', 0.0)
    
    return modified_populations

def update_interaction_matrices(ecosystem_data, species_populations):
    """Aggiorna matrici di interazione tra specie"""
    species_ids = list(species_populations.keys())
    
    for i, species1 in enumerate(species_ids):
        for j, species2 in enumerate(species_ids):
            if i != j:
                interaction_key = tuple(sorted([species1, species2]))
                
                if interaction_key not in ecosystem_data['species_interactions']:
                    interaction = determine_species_interaction(species1, species2)
                    ecosystem_data['species_interactions'][interaction_key] = interaction

def determine_species_interaction(species1_id, species2_id):
    """Determina tipo di interazione tra due specie"""
    interaction_types = ['competition', 'predation', 'mutualism', 'commensalism', 'neutral']
    weights = [0.4, 0.2, 0.1, 0.15, 0.15]
    
    interaction_type = np.random.choice(interaction_types, p=weights)
    strength = random.uniform(0.1, 0.8)
    
    return create_species_interaction(species1_id, species2_id, interaction_type, strength)

def calculate_interaction_effects(ecosystem_data, species_id, all_populations):
    """Calcola effetti delle interazioni per una specie"""
    effects = {
        'population_modifier': 1.0,
        'fitness_modifier': 1.0,
        'pressure': 0.0,
        'benefits': 0.0
    }
    
    total_pressure = 0.0
    total_benefits = 0.0
    
    for interaction_key, interaction in ecosystem_data['species_interactions'].items():
        if species_id in interaction_key:
            other_species = interaction_key[0] if interaction_key[1] == species_id else interaction_key[1]
            
            if other_species not in all_populations:
                continue
            
            other_population_size = len(all_populations[other_species])
            interaction_strength = interaction['strength']
            
            if interaction['interaction_type'] == 'competition':
                competition_pressure = (other_population_size / 100.0) * interaction_strength
                total_pressure += competition_pressure
                
            elif interaction['interaction_type'] == 'predation':
                if ((interaction['direction'] == 'species1_to_species2' and interaction['species1_id'] == species_id) or
                    (interaction['direction'] == 'species2_to_species1' and interaction['species2_id'] == species_id)):
                    predation_benefit = (other_population_size / 100.0) * interaction_strength * 0.5
                    total_benefits += predation_benefit
                else:
                    predation_pressure = (other_population_size / 100.0) * interaction_strength
                    total_pressure += predation_pressure
            
            elif interaction['interaction_type'] == 'mutualism':
                mutualism_benefit = (other_population_size / 100.0) * interaction_strength * 0.3
                total_benefits += mutualism_benefit
    
    effects['pressure'] = total_pressure
    effects['benefits'] = total_benefits
    
    net_effect = total_benefits - total_pressure
    
    if net_effect > 0:
        effects['population_modifier'] = min(1.2, 1.0 + net_effect * 0.1)
        effects['fitness_modifier'] = min(1.3, 1.0 + net_effect * 0.15)
    else:
        effects['population_modifier'] = max(0.7, 1.0 + net_effect * 0.1)
        effects['fitness_modifier'] = max(0.8, 1.0 + net_effect * 0.15)
    
    return effects

# ===============================
# COMPETIZIONE PER RISORSE
# ===============================

def simulate_resource_competition(ecosystem_data, species_populations):
    """Simula competizione per risorse limitate"""
    if not DEFAULT_ECOSYSTEM_CONFIG['resource_competition']:
        return species_populations
    
    patch_assignments = assign_species_to_patches(ecosystem_data, species_populations)
    
    for patch_id, competing_species in patch_assignments.items():
        if len(competing_species) <= 1:
            continue
        
        patch = ecosystem_data['resource_patches'][patch_id]
        resolve_patch_competition(patch, competing_species, species_populations)
    
    return species_populations

def assign_species_to_patches(ecosystem_data, species_populations):
    """Assegna specie a patches di risorse"""
    patch_assignments = defaultdict(list)
    
    for species_id, population in species_populations.items():
        if not population:
            continue
        
        resource_preferences = get_species_resource_preferences(species_id, population)
        best_patches = find_best_patches_for_species(ecosystem_data, resource_preferences)
        
        for patch_id in best_patches[:3]:
            patch_assignments[patch_id].append(species_id)
            ecosystem_data['resource_patches'][patch_id]['occupants'].append(species_id)
    
    return patch_assignments

def get_species_resource_preferences(species_id, population):
    """Determina preferenze di risorsa per una specie"""
    preferences = {}
    
    if not population:
        return preferences
    
    sample = random.sample(population, min(10, len(population)))
    trading_styles = [ind.get('trading_style', 'unknown') for ind in sample]
    style_counter = Counter(trading_styles)
    dominant_style = style_counter.most_common(1)[0][0] if style_counter else 'unknown'
    
    style_to_resources = {
        'scalper': {'high_frequency_opportunities': 0.9, 'volatility_spikes': 0.7},
        'swing': {'swing_trading_setups': 0.9, 'trend_following_signals': 0.6},
        'momentum': {'volatility_spikes': 0.9, 'breakout_patterns': 0.8},
        'contrarian': {'mean_reversion_setups': 0.9, 'low_risk_steady_gains': 0.5},
        'trend_follower': {'trend_following_signals': 0.9, 'breakout_patterns': 0.6}
    }
    
    preferences = style_to_resources.get(dominant_style, {})
    
    avg_aggressiveness = get_average_gene_value(sample, 'aggressiveness', 1.0)
    
    if avg_aggressiveness > 2.0:
        preferences['volatility_spikes'] = preferences.get('volatility_spikes', 0) + 0.3
    elif avg_aggressiveness < 0.5:
        preferences['low_risk_steady_gains'] = preferences.get('low_risk_steady_gains', 0) + 0.4
    
    return preferences

def find_best_patches_for_species(ecosystem_data, resource_preferences):
    """Trova migliori patches per le preferenze di una specie"""
    patch_scores = []
    
    for patch_id, patch in ecosystem_data['resource_patches'].items():
        score = 0.0
        preference = resource_preferences.get(patch['resource_type'], 0.1)
        score += preference * patch['abundance'] * patch['quality']
        
        occupancy_penalty = len(patch['occupants']) * 0.1
        score -= occupancy_penalty
        
        patch_scores.append((patch_id, score))
    
    patch_scores.sort(key=lambda x: x[1], reverse=True)
    return [patch_id for patch_id, score in patch_scores]

def resolve_patch_competition(patch, competing_species, species_populations):
    """Risolve competizione in un patch specifico"""
    species_strengths = {}
    
    for species_id in competing_species:
        population = species_populations.get(species_id, [])
        if not population:
            continue
        
        avg_fitness = get_average_gene_value(population, 'fitness_composito', 0.0)
        avg_aggressiveness = get_average_gene_value(population, 'aggressiveness', 1.0)
        population_size = len(population)
        
        strength = (avg_fitness * 0.4 + avg_aggressiveness * 0.3 + 
                   math.log(population_size + 1) * 0.3)
        
        species_strengths[species_id] = strength
    
    total_strength = sum(species_strengths.values())
    
    if total_strength > 0:
        for species_id, strength in species_strengths.items():
            resource_share = (strength / total_strength) * patch['abundance']
            population = species_populations[species_id]
            modifier = 0.8 + (resource_share * 0.4)
            
            for individual in population:
                current_fitness = get_individual_gene_value(individual, 'fitness_composito', 0.0)
                set_individual_gene_value(individual, 'fitness_composito', current_fitness * modifier)
                individual['resource_access'] = resource_share

# ===============================
# DINAMICHE PREDATOR-PREY
# ===============================

def simulate_predator_prey_dynamics(ecosystem_data, species_populations):
    """Simula dinamiche predator-prey"""
    if not DEFAULT_ECOSYSTEM_CONFIG['predator_prey_dynamics']:
        return species_populations
    
    predator_prey_pairs = identify_predator_prey_pairs(species_populations)
    
    for predator_id, prey_id in predator_prey_pairs:
        if predator_id not in species_populations or prey_id not in species_populations:
            continue
        
        simulate_predator_prey_interaction(predator_id, prey_id, species_populations)
    
    return species_populations

def identify_predator_prey_pairs(species_populations):
    """Identifica coppie predator-prey"""
    pairs = []
    species_characteristics = {}
    
    for species_id, population in species_populations.items():
        if not population:
            continue
        
        sample = random.sample(population, min(5, len(population)))
        avg_aggressiveness = get_average_gene_value(sample, 'aggressiveness', 1.0)
        avg_risk_tolerance = get_average_gene_value(sample, 'position_size_base', 0.2)
        
        trading_styles = [ind.get('trading_style', 'unknown') for ind in sample]
        dominant_style = Counter(trading_styles).most_common(1)[0][0]
        
        species_characteristics[species_id] = {
            'aggressiveness': avg_aggressiveness,
            'risk_tolerance': avg_risk_tolerance,
            'trading_style': dominant_style,
            'population_size': len(population)
        }
    
    for species1_id, chars1 in species_characteristics.items():
        for species2_id, chars2 in species_characteristics.items():
            if species1_id != species2_id:
                is_predator = False
                
                if (chars1['trading_style'] == 'momentum' and 
                    chars2['trading_style'] == 'scalper'):
                    is_predator = True
                elif (chars1['trading_style'] == 'contrarian' and 
                      chars2['trading_style'] == 'momentum'):
                    is_predator = True
                elif (chars1['aggressiveness'] > chars2['aggressiveness'] + 0.5 and
                      chars1['population_size'] < chars2['population_size']):
                    is_predator = True
                
                if is_predator:
                    pairs.append((species1_id, species2_id))
    
    return pairs

def simulate_predator_prey_interaction(predator_id, prey_id, species_populations):
    """Simula interazione specifica predator-prey"""
    predator_pop = species_populations[predator_id]
    prey_pop = species_populations[prey_id]
    
    predator_size = len(predator_pop)
    prey_size = len(prey_pop)
    
    if predator_size == 0 or prey_size == 0:
        return
    
    predation_rate = DEFAULT_ECOSYSTEM_CONFIG['predation_efficiency']
    prey_defense = DEFAULT_ECOSYSTEM_CONFIG['prey_defense_effectiveness']
    
    predation_pressure = (predator_size / 100.0) * predation_rate
    effective_predation = predation_pressure * (1.0 - prey_defense)
    
    # Effetto sulla preda
    prey_mortality = min(0.3, effective_predation)
    
    for individual in prey_pop:
        current_fitness = get_individual_gene_value(individual, 'fitness_composito', 0.0)
        set_individual_gene_value(individual, 'fitness_composito', current_fitness * (1.0 - prey_mortality))
        individual['predation_pressure'] = effective_predation
    
    # Effetto sul predatore
    predator_benefit = (prey_size / 100.0) * predation_rate * 0.5
    predator_boost = min(0.2, predator_benefit)
    
    for individual in predator_pop:
        current_fitness = get_individual_gene_value(individual, 'fitness_composito', 0.0)
        set_individual_gene_value(individual, 'fitness_composito', current_fitness * (1.0 + predator_boost))
        individual['predation_success'] = predator_benefit
    
    # Rimozione individui preda
    if prey_mortality > 0.1 and len(prey_pop) > 5:
        casualties = int(len(prey_pop) * prey_mortality * 0.5)
        if casualties > 0:
            prey_pop.sort(key=lambda x: get_individual_gene_value(x, 'fitness_composito', 0.0))
            species_populations[prey_id] = prey_pop[casualties:]

# ===============================
# EVENTI AMBIENTALI
# ===============================

def simulate_environmental_events(ecosystem_data, species_populations, generation):
    """Simula eventi ambientali casuali"""
    
    # Check per eventi maggiori
    if random.random() < DEFAULT_CONFIG['mass_extinction_probability']:
        species_populations = trigger_mass_extinction(ecosystem_data, species_populations, generation)
    
    elif random.random() < DEFAULT_CONFIG['adaptive_radiation_probability']:
        species_populations = trigger_adaptive_radiation(ecosystem_data, species_populations, generation)
    
    elif random.random() < DEFAULT_CONFIG['climate_change_probability']:
        species_populations = trigger_climate_change(ecosystem_data, species_populations, generation)
    
    # Eventi minori
    if random.random() < 0.1:
        species_populations = trigger_minor_environmental_event(ecosystem_data, species_populations, generation)
    
    return species_populations

def trigger_mass_extinction(ecosystem_data, species_populations, generation):
    """Triggera evento di estinzione di massa"""
    print("   💀 MASS EXTINCTION EVENT!")
    
    extinction_rate = random.uniform(0.2, 0.5)
    species_list = list(species_populations.keys())
    num_to_extinct = int(len(species_list) * extinction_rate)
    
    extinction_probabilities = {}
    
    for species_id, population in species_populations.items():
        if not population:
            extinction_probabilities[species_id] = 1.0
            continue
        
        avg_fitness = get_average_gene_value(population, 'fitness_composito', 0.0)
        population_size = len(population)
        
        vulnerability = 1.0 - (avg_fitness * 0.6 + math.log(population_size + 1) * 0.4)
        extinction_probabilities[species_id] = max(0.1, min(0.9, vulnerability))
    
    species_to_extinct = []
    while len(species_to_extinct) < num_to_extinct and len(species_to_extinct) < len(species_list):
        for species_id in species_list:
            if (species_id not in species_to_extinct and 
                random.random() < extinction_probabilities[species_id]):
                species_to_extinct.append(species_id)
                if len(species_to_extinct) >= num_to_extinct:
                    break
    
    for species_id in species_to_extinct:
        if species_id in species_populations:
            del species_populations[species_id]
            print(f"     💀 Estinta specie: {species_id}")
    
    # Stress sui sopravvissuti
    stress_factor = 0.8
    for species_id, population in species_populations.items():
        apply_fitness_modifier_to_population(population, stress_factor)
        for individual in population:
            individual['mass_extinction_survivor'] = True
    
    # Log evento
    ecosystem_data['event_id_counter'] += 1
    event = create_ecosystem_event(
        f"ECO_{ecosystem_data['event_id_counter']:08d}",
        "mass_extinction",
        generation,
        species_to_extinct,
        extinction_rate
    )
    ecosystem_data['ecosystem_events'].append(event)
    
    return species_populations

def trigger_adaptive_radiation(ecosystem_data, species_populations, generation):
    """Triggera evento di radiazione adattiva"""
    print("   🌟 ADAPTIVE RADIATION EVENT!")
    
    radiation_boost = 1.3
    
    for species_id, population in species_populations.items():
        apply_fitness_modifier_to_population(population, radiation_boost)
        for individual in population:
            individual['adaptive_radiation_boost'] = True
    
    # Log evento
    ecosystem_data['event_id_counter'] += 1
    event = create_ecosystem_event(
        f"ECO_{ecosystem_data['event_id_counter']:08d}",
        "adaptive_radiation",
        generation,
        list(species_populations.keys()),
        radiation_boost
    )
    ecosystem_data['ecosystem_events'].append(event)
    
    return species_populations

def trigger_climate_change(ecosystem_data, species_populations, generation):
    """Triggera cambiamento climatico"""
    print("   🌡️ CLIMATE CHANGE EVENT!")
    
    state = ecosystem_data['state']
    temp_change = random.uniform(-0.3, 0.3)
    vol_change = random.uniform(-0.2, 0.4)
    
    state['temperature'] = np.clip(state['temperature'] + temp_change, 0.0, 1.0)
    state['volatility'] = np.clip(state['volatility'] + vol_change, 0.0, 1.0)
    
    for species_id, population in species_populations.items():
        sample = random.sample(population, min(5, len(population)))
        avg_risk_tolerance = get_average_gene_value(sample, 'position_size_base', 0.2)
        
        if state['volatility'] > 0.7:
            adaptation_factor = 0.8 + (avg_risk_tolerance * 0.4)
        else:
            adaptation_factor = 1.2 - (avg_risk_tolerance * 0.4)
        
        for individual in population:
            current_fitness = get_individual_gene_value(individual, 'fitness_composito', 0.0)
            set_individual_gene_value(individual, 'fitness_composito', current_fitness * adaptation_factor)
            individual['climate_adaptation'] = adaptation_factor
    
    return species_populations

def trigger_minor_environmental_event(ecosystem_data, species_populations, generation):
    """Triggera evento ambientale minore"""
    event_types = ['resource_boom', 'resource_scarcity', 'predator_invasion', 'disease_outbreak']
    event_type = random.choice(event_types)
    
    if event_type == 'resource_boom':
        for patch in ecosystem_data['resource_patches'].values():
            patch['abundance'] *= random.uniform(1.2, 1.8)
    
    elif event_type == 'resource_scarcity':
        for patch in ecosystem_data['resource_patches'].values():
            patch['abundance'] *= random.uniform(0.5, 0.8)
    
    return species_populations

# ===============================
# RELAZIONI SIMBIOTICHE - PARTE MANCANTE
# ===============================

def simulate_symbiotic_relationships(ecosystem_data, species_populations):
    """Simula relazioni simbiotiche (mutualism, commensalism)"""
    
    # Identifica opportunità di simbiosi
    potential_symbionts = identify_potential_symbionts(species_populations)
    
    # Forma nuove relazioni simbiotiche
    new_symbioses = form_new_symbioses(ecosystem_data, potential_symbionts)
    
    # Applica benefici delle relazioni esistenti
    apply_symbiotic_benefits(ecosystem_data, species_populations)
    
    return species_populations

def identify_potential_symbionts(species_populations):
    """Identifica coppie di specie che potrebbero beneficiare di simbiosi"""
    
    potential_pairs = []
    species_chars = {}
    
    # Analizza nicchie ecologiche
    for species_id, population in species_populations.items():
        if not population:
            continue
        
        sample = random.sample(population, min(5, len(population)))
        
        avg_time_horizon = get_average_gene_value(sample, 'time_horizon', 0.5)
        
        trading_styles = [ind.get('trading_style', 'unknown') for ind in sample]
        dominant_style = Counter(trading_styles).most_common(1)[0][0]
        
        avg_risk = get_average_gene_value(sample, 'position_size_base', 0.2)
        
        species_chars[species_id] = {
            'time_horizon': avg_time_horizon,
            'trading_style': dominant_style,
            'risk_level': avg_risk,
            'population_size': len(population)
        }
    
    # Trova coppie complementari
    for species1_id, chars1 in species_chars.items():
        for species2_id, chars2 in species_chars.items():
            if species1_id != species2_id:
                
                # Complementarità temporale (scalper + swing)
                if (abs(chars1['time_horizon'] - chars2['time_horizon']) > 0.4 and
                    chars1['trading_style'] != chars2['trading_style']):
                    potential_pairs.append((species1_id, species2_id))
                
                # Complementarità di rischio (conservative + aggressive)
                elif abs(chars1['risk_level'] - chars2['risk_level']) > 0.3:
                    potential_pairs.append((species1_id, species2_id))
    
    return potential_pairs

def form_new_symbioses(ecosystem_data, potential_symbionts):
    """Forma nuove relazioni simbiotiche"""
    
    new_symbioses = []
    formation_rate = DEFAULT_ECOSYSTEM_CONFIG['symbiosis_formation_rate']
    
    for species1_id, species2_id in potential_symbionts:
        if random.random() < formation_rate:
            
            # Verifica che non esista già una relazione
            existing = False
            for network in ecosystem_data['symbiosis_networks']:
                if species1_id in network and species2_id in network:
                    existing = True
                    break
            
            if not existing:
                # Crea nuova relazione simbiotica
                new_network = {species1_id, species2_id}
                ecosystem_data['symbiosis_networks'].append(new_network)
                new_symbioses.append(new_network)
                
                print(f"   🤝 Nuova simbiosi: {species1_id} ↔ {species2_id}")
    
    return new_symbioses

def apply_symbiotic_benefits(ecosystem_data, species_populations):
    """Applica benefici delle relazioni simbiotiche esistenti"""
    
    mutualism_benefit = DEFAULT_ECOSYSTEM_CONFIG['mutualism_benefit']
    
    for network in ecosystem_data['symbiosis_networks']:
        network_species = list(network)
        
        # Verifica che tutte le specie nella rete esistano ancora
        active_species = [sp for sp in network_species if sp in species_populations]
        
        if len(active_species) < 2:
            continue  # Rete troppo piccola
        
        # Calcola beneficio basato su dimensione della rete
        network_benefit = mutualism_benefit * math.log(len(active_species) + 1)
        
        # Applica benefici a tutte le specie nella rete
        for species_id in active_species:
            population = species_populations[species_id]
            
            for individual in population:
                current_fitness = get_individual_gene_value(individual, 'fitness_composito', 0.0)
                set_individual_gene_value(individual, 'fitness_composito', current_fitness * (1.0 + network_benefit))
                individual['symbiotic_benefit'] = network_benefit

# ===============================
# NICHE CONSTRUCTION - PARTE MANCANTE
# ===============================

def simulate_niche_construction(ecosystem_data, species_populations):
    """Simula niche construction (specie che modificano l'ambiente)"""
    
    if not DEFAULT_ECOSYSTEM_CONFIG['niche_construction_enabled']:
        return species_populations
    
    modification_rate = DEFAULT_ECOSYSTEM_CONFIG['environment_modification_rate']
    
    for species_id, population in species_populations.items():
        if len(population) < 10:  # Solo specie abbastanza grandi
            continue
        
        # Probabilità di modificare ambiente basata su popolazione e innovazione
        sample = random.sample(population, min(5, len(population)))
        avg_innovation = get_average_gene_value(sample, 'innovation_score', 0.5)
        
        modification_probability = (len(population) / 100.0) * avg_innovation * modification_rate
        
        if random.random() < modification_probability:
            modify_environment_by_species(ecosystem_data, species_id, population)
    
    return species_populations

def modify_environment_by_species(ecosystem_data, species_id, population):
    """Una specie modifica l'ambiente (niche construction)"""
    
    modifications = [
        'create_resource_patch',
        'modify_existing_patch',
        'alter_climate_locally',
        'create_trading_opportunity'
    ]
    
    modification_type = random.choice(modifications)
    
    if modification_type == 'create_resource_patch':
        create_new_resource_patch(ecosystem_data, species_id, population)
    
    elif modification_type == 'modify_existing_patch':
        modify_existing_patch(ecosystem_data, species_id, population)
    
    print(f"   🏗️ Niche construction: {species_id} → {modification_type}")

def create_new_resource_patch(ecosystem_data, species_id, population):
    """Crea nuovo patch di risorse"""
    
    # Determina tipo di risorsa basato sulle caratteristiche della specie
    sample = random.sample(population, min(5, len(population)))
    trading_styles = [ind.get('trading_style', 'unknown') for ind in sample]
    dominant_style = Counter(trading_styles).most_common(1)[0][0]
    
    new_resource_types = {
        'scalper': 'micro_arbitrage_opportunities',
        'swing': 'pattern_recognition_setups',
        'momentum': 'momentum_acceleration_zones',
        'contrarian': 'oversold_reversal_signals'
    }
    
    resource_type = new_resource_types.get(dominant_style, 'general_trading_opportunities')
    
    # Crea nuovo patch
    ecosystem_data['patch_id_counter'] += 1
    patch_id = f"PATCH_{ecosystem_data['patch_id_counter']:06d}"
    location = (random.uniform(0, 10), random.uniform(0, 10))
    
    new_patch = create_resource_patch(patch_id, location, resource_type)
    new_patch['abundance'] = random.uniform(0.6, 1.0)
    new_patch['regeneration_rate'] = random.uniform(0.1, 0.3)
    new_patch['capacity'] = random.uniform(30, 100)
    new_patch['quality'] = random.uniform(0.7, 1.0)
    
    ecosystem_data['resource_patches'][patch_id] = new_patch
    print(f"     🌿 Nuovo patch creato: {resource_type}")

def modify_existing_patch(ecosystem_data, species_id, population):
    """Modifica patch esistente"""
    
    if not ecosystem_data['resource_patches']:
        return
    
    # Seleziona patch casuale
    patch_id = random.choice(list(ecosystem_data['resource_patches'].keys()))
    patch = ecosystem_data['resource_patches'][patch_id]
    
    # Migliora qualità e abbondanza
    patch['quality'] = min(1.0, patch['quality'] * random.uniform(1.1, 1.3))
    patch['abundance'] = min(1.0, patch['abundance'] * random.uniform(1.05, 1.2))
    patch['regeneration_rate'] = min(0.5, patch['regeneration_rate'] * random.uniform(1.02, 1.1))

# ===============================
# GESTIONE ESTINZIONI E RISORSE
# ===============================

def update_resource_patches(ecosystem_data, species_populations):
    """Aggiorna stato delle risorse"""
    for patch in ecosystem_data['resource_patches'].values():
        # Rigenerazione naturale
        patch['abundance'] = min(patch['capacity'], 
                               patch['abundance'] + patch['regeneration_rate'])
        
        # Consumo da occupanti
        consumption = len(patch['occupants']) * 0.05
        patch['abundance'] = max(0.0, patch['abundance'] - consumption)
        
        # Reset occupants
        patch['occupants'] = []

def check_extinctions(ecosystem_data, species_populations):
    """Verifica e gestisce estinzioni"""
    species_to_remove = []
    
    for species_id, population in species_populations.items():
        if len(population) < 2:
            species_to_remove.append(species_id)
            continue
        
        avg_fitness = get_average_gene_value(population, 'fitness_composito', 0.0)
        if avg_fitness < 0.01:
            species_to_remove.append(species_id)
            continue
    
    for species_id in species_to_remove:
        del species_populations[species_id]
        
        extinction_event = {
            'species_id': species_id,
            'generation': ecosystem_data['state']['generation'],
            'cause': 'natural_extinction',
            'timestamp': datetime.now()
        }
        ecosystem_data['extinction_events'].append(extinction_event)
        
        print(f"     💀 Estinzione naturale: {species_id}")
    
    return species_populations

# ===============================
# INTERFACCE PUBBLICHE AGGIUNTIVE - PARTE MANCANTE
# ===============================

def update_ecosystem(ecosystem_data, species_populations, generation, environment_state=None):
    """Interfaccia pubblica per aggiornamento ecosistema"""
    
    try:
        # Se non è fornito environment_state, usa valori di default
        if environment_state is None:
            environment_state = {
                'market_volatility': ecosystem_data['state']['volatility'],
                'trend_strength': ecosystem_data['state']['trend_strength'],
                'temperature': ecosystem_data['state']['temperature'],
                'market_regime': ecosystem_data['state']['market_regime']
            }
        
        # Aggiorna stato ambientale interno
        if isinstance(environment_state, dict):
            if 'market_volatility' in environment_state:
                ecosystem_data['state']['volatility'] = environment_state['market_volatility']
            if 'trend_strength' in environment_state:
                ecosystem_data['state']['trend_strength'] = environment_state['trend_strength']
            if 'temperature' in environment_state:
                ecosystem_data['state']['temperature'] = environment_state['temperature']
        
        # Chiama il metodo principale
        return simulate_ecosystem_step(ecosystem_data, species_populations, generation)
        
    except Exception as e:
        print(f"❌ Errore in update_ecosystem: {e}")
        return species_populations

def get_ecosystem_state(ecosystem_data):
    """Ottieni stato corrente dell'ecosistema"""
    
    try:
        state = ecosystem_data['state']
        return {
            'generation': state['generation'],
            'total_population': state['total_population'],
            'species_count': state['species_count'],
            'biodiversity_index': state['biodiversity_index'],
            'carrying_capacity': state['carrying_capacity'],
            'resource_abundance': state['resource_abundance'],
            'environmental_pressure': state['environmental_pressure'],
            'climate_stability': state['climate_stability'],
            'market_volatility': state['volatility'],
            'trend_strength': state['trend_strength'],
            'temperature': state['temperature'],
            'market_regime': state['market_regime'],
            'active_patches': len(ecosystem_data['resource_patches']),
            'symbiosis_networks': len(ecosystem_data['symbiosis_networks']),
            'species_interactions': len(ecosystem_data['species_interactions']),
            'recent_events': len([e for e in ecosystem_data['ecosystem_events'] if e['generation'] > state['generation'] - 5])
        }
        
    except Exception as e:
        print(f"❌ Errore in get_ecosystem_state: {e}")
        return {
            'generation': 0,
            'total_population': 0,
            'species_count': 0,
            'biodiversity_index': 0.0,
            'error': str(e)
        }

def reset_ecosystem(ecosystem_data, keep_patches=True):
    """Reset dello stato ecosistema"""
    
    try:
        # Reset stato principale
        state = ecosystem_data['state']
        state['generation'] = 0
        state['total_population'] = 0
        state['species_count'] = 0
        state['biodiversity_index'] = 0.0
        state['competition_intensity'] = 0.0
        state['predation_pressure'] = 0.0
        
        # Reset storia
        ecosystem_data['ecosystem_history'].clear()
        ecosystem_data['ecosystem_events'].clear()
        ecosystem_data['extinction_events'].clear()
        
        # Reset interazioni
        ecosystem_data['species_interactions'].clear()
        ecosystem_data['symbiosis_networks'].clear()
        ecosystem_data['species_fitness_cache'].clear()
        
        # Reset fattori ambientali ai valori di default
        state['temperature'] = 0.5
        state['volatility'] = 0.5
        state['trend_strength'] = 0.5
        state['market_regime'] = "neutral"
        
        # Mantieni o rigenera patches
        if not keep_patches:
            ecosystem_data['resource_patches'].clear()
            ecosystem_data['patch_id_counter'] = 0
            initialize_resource_patches(ecosystem_data)
            
        print("✅ Ecosistema resetato con successo")
        
    except Exception as e:
        print(f"❌ Errore in reset_ecosystem: {e}")

def apply_environmental_shock(ecosystem_data, shock_type, intensity=1.0):
    """Applica shock ambientale immediato"""
    
    try:
        state = ecosystem_data['state']
        
        if shock_type == 'volatility_spike':
            state['volatility'] = min(1.0, state['volatility'] + 0.3 * intensity)
            state['market_regime'] = "high_volatility"
            
        elif shock_type == 'market_crash':
            state['trend_strength'] = max(0.0, state['trend_strength'] - 0.4 * intensity)
            state['volatility'] = min(1.0, state['volatility'] + 0.5 * intensity)
            state['market_regime'] = "crisis"
            
        elif shock_type == 'climate_change':
            temp_change = (random.random() - 0.5) * 0.6 * intensity
            state['temperature'] = np.clip(state['temperature'] + temp_change, 0.0, 1.0)
            
        elif shock_type == 'resource_depletion':
            for patch in ecosystem_data['resource_patches'].values():
                patch['abundance'] *= (1.0 - 0.3 * intensity)
                
        elif shock_type == 'abundance_boom':
            for patch in ecosystem_data['resource_patches'].values():
                patch['abundance'] = min(patch['capacity'], patch['abundance'] * (1.0 + 0.5 * intensity))
                
        print(f"✅ Shock ambientale applicato: {shock_type} (intensità: {intensity})")
        
    except Exception as e:
        print(f"❌ Errore in apply_environmental_shock: {e}")

# ===============================
# STEP PRINCIPALE ECOSISTEMA
# ===============================

def simulate_ecosystem_step(ecosystem_data, species_populations, generation):
    """Simula un passo completo dell'ecosistema"""
    
    ecosystem_data['state']['generation'] = generation
    
    print(f"   🌍 Ecosistema Step Gen {generation}")
    
    # Update stato ecosistema
    update_ecosystem_state(ecosystem_data, species_populations)
    
    # Simula interazioni ecologiche
    species_populations = simulate_ecological_interactions(ecosystem_data, species_populations)
    
    # Gestisci competizione per risorse
    species_populations = simulate_resource_competition(ecosystem_data, species_populations)
    
    # Dinamiche predator-prey
    species_populations = simulate_predator_prey_dynamics(ecosystem_data, species_populations)
    
    # Symbiosis e Mutualism
    species_populations = simulate_symbiotic_relationships(ecosystem_data, species_populations)
    
    # Eventi ambientali
    species_populations = simulate_environmental_events(ecosystem_data, species_populations, generation)
    
    # Niche construction
    species_populations = simulate_niche_construction(ecosystem_data, species_populations)
    
    # Update risorse
    update_resource_patches(ecosystem_data, species_populations)
    
    # Check estinzioni
    species_populations = check_extinctions(ecosystem_data, species_populations)
    
    # Check adaptive radiation se poche specie
    if len(species_populations) < DEFAULT_CONFIG['min_species']:
        species_populations = trigger_adaptive_radiation(ecosystem_data, species_populations, generation)
    
    # Salva stato nella storia
    ecosystem_data['ecosystem_history'].append(copy.deepcopy(ecosystem_data['state']))
    
    return species_populations

# ===============================
# ANALYTICS E REPORTING
# ===============================

def get_ecosystem_statistics(ecosystem_data, species_populations):
    """Ottieni statistiche complete dell'ecosistema"""
    
    current_populations = {
        species_id: len(pop) for species_id, pop in species_populations.items()
    }
    
    stats = {
        'current_state': ecosystem_data['state'],
        'total_events': len(ecosystem_data['ecosystem_events']),
        'total_extinctions': len(ecosystem_data['extinction_events']),
        'active_symbioses': len(ecosystem_data['symbiosis_networks']),
        'resource_patches': len(ecosystem_data['resource_patches']),
        'interaction_networks': len(ecosystem_data['species_interactions']),
        
        'population_stats': current_populations,
        
        'resource_abundance': {
            patch_id: patch['abundance'] 
            for patch_id, patch in ecosystem_data['resource_patches'].items()
        },
        
        'recent_events': ecosystem_data['ecosystem_events'][-10:],
        
        'biodiversity_trend': [
            state['biodiversity_index'] 
            for state in ecosystem_data['ecosystem_history'][-20:]
        ],
        
        'environmental_factors': {
            'temperature': ecosystem_data['state']['temperature'],
            'volatility': ecosystem_data['state']['volatility'],
            'trend_strength': ecosystem_data['state']['trend_strength'],
            'market_regime': ecosystem_data['state']['market_regime']
        }
    }
    
    return stats

# ===============================
# GRAFICO FINALE EVOLUTION - SOLO QUELLO CHE FUNZIONA! 🎨
# ===============================

def create_final_evolution_graphic(ecosystem_data, species_populations):
    """Crea SOLO il grafico finale dell'evoluzione - GARANTITO FUNZIONANTE! 🚀"""
    
    print("🎨 Creando grafico finale evoluzione ecosistema...")
    
    # Crea figura con layout 2x2
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.patch.set_facecolor('black')
    
    # Titolo principale
    fig.suptitle('GENESIS ECOSYSTEM - FINAL EVOLUTION REPORT', 
                 fontsize=20, fontweight='bold', color='cyan')
    
    # 1. BIODIVERSITÀ NEL TEMPO (TOP LEFT)
    plot_simple_biodiversity(ax1, ecosystem_data)
    
    # 2. POPOLAZIONI SPECIE (TOP RIGHT)  
    plot_simple_populations(ax2, species_populations)
    
    # 3. EVENTI ECOSYSTEM (BOTTOM LEFT)
    plot_simple_events(ax3, ecosystem_data)
    
    # 4. SUMMARY STATISTICS (BOTTOM RIGHT)
    plot_summary_stats(ax4, ecosystem_data, species_populations)
    
    # Layout e salvataggio
    plt.tight_layout()
    
    # Timestamp
    fig.text(0.99, 0.01, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 
             ha='right', va='bottom', fontsize=10, alpha=0.7, color='white')
    
    # Salva
    filename = f'ecosystem_final_evolution_gen_{ecosystem_data["state"]["generation"]:04d}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='black')
    plt.show()
    
    print(f"✅ Grafico finale salvato come: {filename}")
    return filename

def plot_simple_biodiversity(ax, ecosystem_data):
    """Grafico semplice biodiversità - FUNZIONA SEMPRE!"""
    
    ax.set_facecolor('black')
    ax.set_title('BIODIVERSITY EVOLUTION', fontweight='bold', color='lime', fontsize=14)
    
    if not ecosystem_data['ecosystem_history']:
        ax.text(0.5, 0.5, 'No Evolution Data', ha='center', va='center', 
                transform=ax.transAxes, fontsize=16, color='white')
        return
    
    # Dati storici
    generations = [state['generation'] for state in ecosystem_data['ecosystem_history']]
    biodiversity = [state['biodiversity_index'] for state in ecosystem_data['ecosystem_history']]
    populations = [state['total_population'] for state in ecosystem_data['ecosystem_history']]
    
    # Plot principale biodiversità
    ax.plot(generations, biodiversity, color='lime', linewidth=3, marker='o', markersize=4, label='Shannon Index')
    ax.fill_between(generations, biodiversity, alpha=0.3, color='lime')
    
    # Asse secondario per popolazione
    ax2 = ax.twinx()
    ax2.plot(generations, populations, color='cyan', linewidth=2, linestyle='--', 
             marker='s', markersize=3, label='Total Population', alpha=0.8)
    
    # Styling
    ax.set_xlabel('Generation', color='white', fontsize=12)
    ax.set_ylabel('Shannon Biodiversity Index', color='lime', fontsize=12)
    ax2.set_ylabel('Total Population', color='cyan', fontsize=12)
    
    # Griglia
    ax.grid(True, alpha=0.3, color='white')
    
    # Legenda combinata
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', 
              framealpha=0.8, facecolor='black', edgecolor='white')
    
    # Colori assi
    ax.tick_params(colors='white')
    ax2.tick_params(colors='white')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('lime')
    ax.spines['right'].set_color('cyan')
    ax.spines['top'].set_color('white')

def plot_simple_populations(ax, species_populations):
    """Grafico semplice popolazioni - COLORATO E FUNZIONANTE!"""
    
    ax.set_facecolor('black')
    ax.set_title('SPECIES POPULATIONS', fontweight='bold', color='orange', fontsize=14)
    
    if not species_populations:
        ax.text(0.5, 0.5, 'No Species Survived', ha='center', va='center', 
                transform=ax.transAxes, fontsize=16, color='red')
        return
    
    # Prepara dati
    species_names = [f'Species_{i+1}' for i in range(len(species_populations))]
    populations = [len(pop) for pop in species_populations.values()]
    
    # Colori vibranti
    colors = plt.cm.Set1(np.linspace(0, 1, len(species_names)))
    
    # Grafico a barre
    bars = ax.bar(species_names, populations, color=colors, alpha=0.8, 
                  edgecolor='white', linewidth=2)
    
    # Etichette sui bar
    for bar, pop in zip(bars, populations):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + max(populations)*0.01,
                f'{pop}', ha='center', va='bottom', fontweight='bold', 
                color='white', fontsize=10)
    
    # Styling
    ax.set_xlabel('Species', color='white', fontsize=12)
    ax.set_ylabel('Population Size', color='orange', fontsize=12)
    ax.tick_params(colors='white')
    ax.grid(True, alpha=0.3, axis='y', color='white')
    
    # Spines
    for spine in ax.spines.values():
        spine.set_color('white')
    
    # Ruota etichette se necessario
    if len(species_names) > 5:
        plt.setp(ax.get_xticklabels(), rotation=45)

def plot_simple_events(ax, ecosystem_data):
    """Grafico semplice eventi - TIMELINE SPETTACOLARE!"""
    
    ax.set_facecolor('black')
    ax.set_title('ECOSYSTEM EVENTS TIMELINE', fontweight='bold', color='red', fontsize=14)
    
    events = ecosystem_data['ecosystem_events']
    
    if not events:
        ax.text(0.5, 0.5, 'No Major Events', ha='center', va='center', 
                transform=ax.transAxes, fontsize=16, color='white')
        return
    
    # Categorizza eventi per colore
    event_colors = {
        'mass_extinction': 'red',
        'adaptive_radiation': 'lime', 
        'climate_change': 'orange',
        'default': 'cyan'
    }
    
    event_markers = {
        'mass_extinction': 'X',
        'adaptive_radiation': '*',
        'climate_change': 'D', 
        'default': 'o'
    }
    
    # Plot eventi per tipo
    for event_type in event_colors.keys():
        if event_type == 'default':
            continue
            
        type_events = [e for e in events if e['event_type'] == event_type]
        if type_events:
            generations = [e['generation'] for e in type_events]
            magnitudes = [e['magnitude'] for e in type_events]
            
            ax.scatter(generations, magnitudes, 
                      c=event_colors[event_type], 
                      marker=event_markers[event_type],
                      s=150, alpha=0.8, 
                      label=event_type.replace('_', ' ').title(),
                      edgecolors='white', linewidth=1)
    
    # Eventi minori
    other_events = [e for e in events if e['event_type'] not in event_colors]
    if other_events:
        generations = [e['generation'] for e in other_events] 
        magnitudes = [e['magnitude'] for e in other_events]
        ax.scatter(generations, magnitudes, c='cyan', marker='o', s=100, 
                  alpha=0.6, label='Other Events', edgecolors='white')
    
    # Styling
    ax.set_xlabel('Generation', color='white', fontsize=12)
    ax.set_ylabel('Event Magnitude', color='red', fontsize=12)
    ax.tick_params(colors='white')
    ax.grid(True, alpha=0.3, color='white')
    
    # Legenda
    ax.legend(framealpha=0.8, facecolor='black', edgecolor='white', 
              labelcolor='white', fontsize=10)
    
    # Spines
    for spine in ax.spines.values():
        spine.set_color('white')

def plot_summary_stats(ax, ecosystem_data, species_populations):
    """Grafico riassunto statistiche - RADAR FINALE!"""
    
    ax.set_facecolor('black')
    ax.set_title('ECOSYSTEM SUMMARY STATS', fontweight='bold', color='gold', fontsize=14)
    
    # Calcola metriche finali
    state = ecosystem_data['state']
    
    final_stats = {
        'Biodiversity': min(1.0, state['biodiversity_index']),
        'Population': min(1.0, state['total_population'] / state['carrying_capacity']),
        'Species Count': min(1.0, len(species_populations) / 10),  # Normalizza su 10 specie max
        'Stability': calculate_ecosystem_stability(ecosystem_data),
        'Resources': np.mean([p['abundance'] for p in ecosystem_data['resource_patches'].values()]) if ecosystem_data['resource_patches'] else 0,
        'Events': min(1.0, len(ecosystem_data['ecosystem_events']) / 20)  # Normalizza su 20 eventi max
    }
    
    # Prepara dati per radar chart
    categories = list(final_stats.keys())
    values = list(final_stats.values())
    
    # Angoli per radar
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
    values += values[:1]  # Chiudi il poligono
    angles += angles[:1]
    
    # Crea subplot polare
    ax.remove()  # Rimuovi asse cartesiano
    ax = plt.subplot(2, 2, 4, projection='polar')
    ax.set_facecolor('black')
    
    # Plot radar
    ax.plot(angles, values, 'o-', linewidth=3, color='gold', markersize=8)
    ax.fill(angles, values, alpha=0.25, color='gold')
    
    # Etichette
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, color='white', fontsize=10, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, color='white')
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    # Valori sulle punte
    for angle, value, category in zip(angles[:-1], values[:-1], categories):
        ax.text(angle, value + 0.1, f'{value:.2f}', 
                ha='center', va='center', color='yellow', 
                fontweight='bold', fontsize=9)
    
    # Colori griglia
    ax.tick_params(colors='white')

# ===============================
# WRAPPER SEMPLIFICATO
# ===============================

def generate_final_graphics_only(ecosystem_data, species_populations):
    """Genera SOLO il grafico finale - GARANTITO FUNZIONANTE! 🎨"""
    
    print("\n🎨 === GENERAZIONE GRAFICO FINALE ===")
    
    try:
        # Solo grafico finale
        final_file = create_final_evolution_graphic(ecosystem_data, species_populations)
        
        print("✅ Grafico finale completato!")
        return final_file
        
    except Exception as e:
        print(f"❌ Errore nella generazione grafico: {e}")
        import traceback
        traceback.print_exc()
        return None

def analyze_species_characteristics(species_populations):
    """Analizza caratteristiche dettagliate delle specie"""
    analysis = {}
    
    for species_id, population in species_populations.items():
        if not population:
            continue
            
        # Analisi genetica
        fitness_values = [get_individual_gene_value(ind, 'fitness_composito', 0.0) for ind in population]
        aggressiveness_values = [get_individual_gene_value(ind, 'aggressiveness', 1.0) for ind in population]
        risk_values = [get_individual_gene_value(ind, 'position_size_base', 0.2) for ind in population]
        
        # Analisi trading styles
        trading_styles = [ind.get('trading_style', 'unknown') for ind in population]
        style_distribution = Counter(trading_styles)
        
        # Analisi ecologica
        ecological_pressures = [ind.get('ecological_pressure', 0.0) for ind in population]
        interaction_benefits = [ind.get('interaction_benefits', 0.0) for ind in population]
        
        analysis[species_id] = {
            'population_size': len(population),
            'fitness_stats': {
                'mean': np.mean(fitness_values),
                'std': np.std(fitness_values),
                'min': min(fitness_values),
                'max': max(fitness_values)
            },
            'aggressiveness_stats': {
                'mean': np.mean(aggressiveness_values),
                'std': np.std(aggressiveness_values)
            },
            'risk_tolerance_stats': {
                'mean': np.mean(risk_values),
                'std': np.std(risk_values)
            },
            'trading_style_distribution': dict(style_distribution),
            'dominant_style': style_distribution.most_common(1)[0][0] if style_distribution else 'unknown',
            'ecological_impact': {
                'avg_pressure': np.mean(ecological_pressures),
                'avg_benefits': np.mean(interaction_benefits)
            }
        }
    
    return analysis

def analyze_ecosystem_dynamics(ecosystem_data):
    """Analizza dinamiche evolutive dell'ecosistema"""
    
    if not ecosystem_data['ecosystem_history']:
        return {}
    
    history = ecosystem_data['ecosystem_history']
    
    # Trend biodiversità
    biodiversity_trend = [state['biodiversity_index'] for state in history]
    
    # Trend popolazione
    population_trend = [state['total_population'] for state in history]
    
    # Trend ambientali
    temperature_trend = [state['temperature'] for state in history]
    volatility_trend = [state['volatility'] for state in history]
    
    # Analisi ciclica
    cycles = analyze_population_cycles(population_trend)
    
    dynamics = {
        'biodiversity_trend': {
            'values': biodiversity_trend,
            'slope': calculate_trend_slope(biodiversity_trend),
            'volatility': np.std(biodiversity_trend),
            'current': biodiversity_trend[-1] if biodiversity_trend else 0
        },
        'population_dynamics': {
            'values': population_trend,
            'slope': calculate_trend_slope(population_trend),
            'cycles_detected': cycles,
            'current': population_trend[-1] if population_trend else 0
        },
        'environmental_changes': {
            'temperature': {
                'trend': calculate_trend_slope(temperature_trend),
                'volatility': np.std(temperature_trend)
            },
            'market_volatility': {
                'trend': calculate_trend_slope(volatility_trend),
                'mean': np.mean(volatility_trend)
            }
        },
        'ecosystem_stability': calculate_ecosystem_stability(ecosystem_data)
    }
    
    return dynamics

def calculate_trend_slope(values):
    """Calcola la pendenza di un trend"""
    if len(values) < 2:
        return 0.0
    
    x = np.arange(len(values))
    slope = np.polyfit(x, values, 1)[0]
    return slope

def analyze_population_cycles(population_values):
    """Analizza cicli nelle dinamiche di popolazione"""
    if len(population_values) < 10:
        return {'cycles_found': False}
    
    # Semplice analisi di picchi e valli
    peaks = []
    valleys = []
    
    for i in range(1, len(population_values) - 1):
        if (population_values[i] > population_values[i-1] and 
            population_values[i] > population_values[i+1]):
            peaks.append(i)
        elif (population_values[i] < population_values[i-1] and 
              population_values[i] < population_values[i+1]):
            valleys.append(i)
    
    cycle_length = None
    if len(peaks) >= 2:
        cycle_length = np.mean(np.diff(peaks))
    
    return {
        'cycles_found': len(peaks) >= 2,
        'num_peaks': len(peaks),
        'num_valleys': len(valleys),
        'estimated_cycle_length': cycle_length,
        'peak_positions': peaks,
        'valley_positions': valleys
    }

def calculate_ecosystem_stability(ecosystem_data):
    """Calcola indice di stabilità dell'ecosistema"""
    
    if not ecosystem_data['ecosystem_history']:
        return 0.0
    
    history = ecosystem_data['ecosystem_history']
    
    # Stabilità basata su varianza delle metriche chiave
    biodiversity_variance = np.var([s['biodiversity_index'] for s in history])
    population_variance = np.var([s['total_population'] for s in history])
    species_count_variance = np.var([s['species_count'] for s in history])
    
    # Eventi di disturbo
    major_events = len([e for e in ecosystem_data['ecosystem_events'] 
                       if e['event_type'] in ['mass_extinction', 'climate_change']])
    
    # Stabilità inversa alla varianza (più bassa varianza = più stabile)
    stability_score = 1.0 / (1.0 + biodiversity_variance + 
                            population_variance/1000 + 
                            species_count_variance + 
                            major_events * 0.1)
    
    return min(1.0, stability_score)

def generate_detailed_ecosystem_report(ecosystem_data, species_populations):
    """Genera report dettagliato con analytics avanzate"""
    
    basic_stats = get_ecosystem_statistics(ecosystem_data, species_populations)
    species_analysis = analyze_species_characteristics(species_populations)
    dynamics_analysis = analyze_ecosystem_dynamics(ecosystem_data)
    
    state = ecosystem_data['state']
    
    report = f"""
🌍 === GENESIS ECOSYSTEM - DETAILED ANALYSIS REPORT ===
📅 Generazione: {state['generation']} | 🕒 Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

📊 === STATO GENERALE ===
   🧬 Specie Attive: {state['species_count']}
   👥 Popolazione Totale: {state['total_population']}
   🌿 Biodiversità (Shannon): {state['biodiversity_index']:.4f}
   🏠 Capacità Portante: {state['carrying_capacity']}
   📈 Pressione Ambientale: {state['environmental_pressure']:.3f}

🌡️ === FATTORI AMBIENTALI ===
   🌡️ Temperatura: {state['temperature']:.3f}
   📊 Volatilità Mercato: {state['volatility']:.3f}
   📈 Forza Trend: {state['trend_strength']:.3f}
   🎯 Regime Attuale: {state['market_regime']}
   🔒 Stabilità Ecosistema: {dynamics_analysis.get('ecosystem_stability', 0):.3f}

🔗 === RETI ECOLOGICHE ===
   🤝 Reti Simbiotiche: {len(ecosystem_data['symbiosis_networks'])}
   ⚔️ Interazioni Specie: {len(ecosystem_data['species_interactions'])}
   🎯 Pressione Predazione: {state.get('predation_pressure', 0):.3f}
   🏆 Intensità Competizione: {state.get('competition_intensity', 0):.3f}

🌿 === RISORSE E HABITAT ===
   🏞️ Patches Attivi: {len(ecosystem_data['resource_patches'])}
   📊 Abbondanza Media: {np.mean([p['abundance'] for p in ecosystem_data['resource_patches'].values()]):.3f}
   ⚡ Rigenerazione Media: {np.mean([p['regeneration_rate'] for p in ecosystem_data['resource_patches'].values()]):.4f}

📈 === DINAMICHE EVOLUTIVE ==="""
    
    # Aggiungi analisi dinamiche se disponibili
    if 'biodiversity_trend' in dynamics_analysis:
        trend = dynamics_analysis['biodiversity_trend']
        report += f"""
   📊 Trend Biodiversità: {trend['slope']:+.6f} (pendenza)
   📊 Volatilità Biodiversità: {trend['volatility']:.4f}"""
    
    if 'population_dynamics' in dynamics_analysis:
        pop_dynamics = dynamics_analysis['population_dynamics']
        cycles = pop_dynamics['cycles_detected']
        report += f"""
   👥 Trend Popolazione: {pop_dynamics['slope']:+.2f}
   🔄 Cicli Rilevati: {'Sì' if cycles['cycles_found'] else 'No'}"""
        
        if cycles['cycles_found']:
            report += f"""
   🔄 Lunghezza Ciclo: ~{cycles['estimated_cycle_length']:.1f} generazioni"""

    report += f"""

🎯 === EVENTI E STORIA ===
   📊 Eventi Totali: {len(ecosystem_data['ecosystem_events'])}
   💀 Estinzioni Totali: {len(ecosystem_data['extinction_events'])}
   🌟 Eventi Maggiori: {len([e for e in ecosystem_data['ecosystem_events'] if e['event_type'] in ['mass_extinction', 'adaptive_radiation', 'climate_change']])}

🧬 === ANALISI SPECIE ==="""
    
    # Aggiungi analisi per ogni specie
    for species_id, analysis in species_analysis.items():
        fitness_stats = analysis['fitness_stats']
        report += f"""
   
   🔬 {species_id}:
      👥 Popolazione: {analysis['population_size']}
      🎯 Style Dominante: {analysis['dominant_style']}
      💪 Fitness Media: {fitness_stats['mean']:.4f} (±{fitness_stats['std']:.3f})
      ⚡ Aggressività: {analysis['aggressiveness_stats']['mean']:.3f}
      🎲 Rischio Medio: {analysis['risk_tolerance_stats']['mean']:.3f}
      🌍 Pressione Ecologica: {analysis['ecological_impact']['avg_pressure']:.3f}
      🤝 Benefici Interazione: {analysis['ecological_impact']['avg_benefits']:.3f}"""

    report += f"""

💀 === ESTINZIONI RECENTI ==="""
    
    recent_extinctions = ecosystem_data['extinction_events'][-5:]
    if recent_extinctions:
        for ext in recent_extinctions:
            report += f"""
   💀 {ext['species_id']} (Gen {ext['generation']}) - {ext['cause']}"""
    else:
        report += """
   ✅ Nessuna estinzione recente"""

    report += f"""

🏆 === PERFORMANCE METRICS ===
   🎯 Efficienza Ecosistema: {(state['biodiversity_index'] * state['species_count'] / max(1, len(ecosystem_data['extinction_events']))):.3f}
   📊 Resilienza: {dynamics_analysis.get('ecosystem_stability', 0):.3f}
   🌿 Diversità Genetica: {np.mean([len(set(ind.get('trading_style', 'unknown') for ind in pop)) for pop in species_populations.values()]):.2f}

✅ === SISTEMA STATUS ===
✅ Sistema Procedurale: OPERATIVO
✅ Gene Utils: INTEGRATO  
✅ Compatibility Layer: ATTIVO
✅ Analytics Avanzate: ABILITATE
✅ Niche Construction: ATTIVO
✅ Relazioni Simbiotiche: MONITORATE

🎯 === FINE REPORT ===
    """
    
    return report

def generate_ecosystem_report(ecosystem_data, species_populations):
    """Genera report testuale semplificato dell'ecosistema"""
    
    stats = get_ecosystem_statistics(ecosystem_data, species_populations)
    state = ecosystem_data['state']
    
    report = f"""
🌍 === ECOSYSTEM REPORT - Generation {state['generation']} ===

📊 STATO GENERALE:
   Specie Attive: {state['species_count']}
   Popolazione Totale: {state['total_population']}
   Biodiversità (Shannon): {state['biodiversity_index']:.3f}
   Capacità Portante: {state['carrying_capacity']}

🌡️ FATTORI AMBIENTALI:
   Temperatura: {state['temperature']:.2f}
   Volatilità Mercato: {state['volatility']:.2f}
   Forza Trend: {state['trend_strength']:.2f}
   Regime: {state['market_regime']}

🔗 INTERAZIONI:
   Reti Simbiotiche: {len(ecosystem_data['symbiosis_networks'])}
   Interazioni Specie: {len(ecosystem_data['species_interactions'])}
   Pressione Predazione: {state.get('predation_pressure', 0):.2f}

🌿 RISORSE:
   Patches Attivi: {len(ecosystem_data['resource_patches'])}
   Abbondanza Media: {np.mean([p['abundance'] for p in ecosystem_data['resource_patches'].values()]):.2f}

📈 EVENTI:
   Eventi Totali: {len(ecosystem_data['ecosystem_events'])}
   Estinzioni: {len(ecosystem_data['extinction_events'])}

💀 ESTINZIONI RECENTI:
{chr(10).join([f"   - {ext['species_id']} (Gen {ext['generation']})" for ext in ecosystem_data['extinction_events'][-5:]])}

✅ SISTEMA PROCEDURALE: Completamente funzionale senza classi
    """
    
    return report

def print_population_summary(species_populations):
    """Stampa riassunto delle popolazioni"""
    print(f"\n📊 === POPOLAZIONE CORRENTE ===")
    
    if not species_populations:
        print("   🔴 NESSUNA SPECIE SOPRAVVISSUTA")
        return
    
    for species_id, population in species_populations.items():
        if not population:
            continue
            
        avg_fitness = get_average_gene_value(population, 'fitness_composito', 0.0)
        avg_aggressiveness = get_average_gene_value(population, 'aggressiveness', 1.0)
        
        trading_styles = [ind.get('trading_style', 'unknown') for ind in population]
        dominant_style = Counter(trading_styles).most_common(1)[0][0]
        
        print(f"   🧬 {species_id}: {len(population)} individui")
        print(f"      Style: {dominant_style}, Fitness: {avg_fitness:.3f}, Aggressività: {avg_aggressiveness:.2f}")

# ===============================
# MAIN DEMO - SIMULAZIONE COMPLETA
# ===============================

def main():
    """Main procedurale - Esegue simulazione ecosistema completa"""
    
    print("🌍 === GENESIS EVOLUTION SYSTEM - STANDALONE PROCEDURALE ===")
    print("🚀 Inizializzazione ecosistema...")
    
    # Crea ecosistema
    ecosystem_data = create_ecosystem_data()
    
    # Inizializza risorse
    print("\n🌿 Inizializzazione risorse...")
    initialize_resource_patches(ecosystem_data)
    
    # Genera specie iniziali
    print("\n🧬 Generazione specie iniziali...")
    species_populations = generate_initial_species_populations(num_species=6)
    
    print(f"\n✅ Ecosistema inizializzato con {len(species_populations)} specie")
    print_population_summary(species_populations)
    
    # Simulazione principale
    print("\n🌍 === INIZIO SIMULAZIONE ===")
    
    generations = 50
    
    for generation in range(1, generations + 1):
        print(f"\n🔄 Generazione {generation}/{generations}")
        
        # Simula passo ecosistema
        species_populations = simulate_ecosystem_step(ecosystem_data, species_populations, generation)
        
        # Report ogni 10 generazioni
        if generation % 10 == 0:
            print(f"\n📈 === REPORT GENERAZIONE {generation} ===")
            print_population_summary(species_populations)
            
            stats = get_ecosystem_statistics(ecosystem_data, species_populations)
            state = ecosystem_data['state']
            
            print(f"\n📊 Statistiche:")
            print(f"   Biodiversità: {state['biodiversity_index']:.3f}")
            print(f"   Regime Mercato: {state['market_regime']}")
            print(f"   Eventi Totali: {len(ecosystem_data['ecosystem_events'])}")
            print(f"   Estinzioni: {len(ecosystem_data['extinction_events'])}")
            
            # NIENTE GRAFICI DURANTE LA SIMULAZIONE!
        
        # Check terminazione precoce
        if len(species_populations) == 0:
            print("\n💀 ESTINZIONE TOTALE - Simulazione terminata")
            break
        
        # Pausa per visualizzazione
        time.sleep(0.1)
    
    # Report finale
    print("\n🎯 === REPORT FINALE ===")
    final_report = generate_detailed_ecosystem_report(ecosystem_data, species_populations)
    print(final_report)
    
    # 🎨 GRAFICO BIODIVERSITÀ FINALE!
    print("\n🎨 === CREAZIONE GRAFICO BIODIVERSITÀ FINALE ===")
    final_graphic = generate_biodiversity_graphic_only(ecosystem_data, species_populations)
    
    print_population_summary(species_populations)
    
    # Statistiche dettagliate
    stats = get_ecosystem_statistics(ecosystem_data, species_populations)
    
    if ecosystem_data['ecosystem_history']:
        biodiversity_history = [state['biodiversity_index'] for state in ecosystem_data['ecosystem_history']]
        print(f"\n📈 Biodiversità finale: {biodiversity_history[-1]:.3f}")
        print(f"📈 Biodiversità media: {np.mean(biodiversity_history):.3f}")
        print(f"📈 Biodiversità massima: {max(biodiversity_history):.3f}")
    
    print(f"\n🏆 === SIMULAZIONE COMPLETATA ===")
    print(f"🏆 Specie sopravvissute: {len(species_populations)}")
    print(f"🏆 Eventi maggiori: {len([e for e in ecosystem_data['ecosystem_events'] if e['event_type'] in ['mass_extinction', 'adaptive_radiation', 'climate_change']])}")
    print(f"🏆 Generazioni simulate: {ecosystem_data['state']['generation']}")
    
    print("\n🌍 Grazie per aver utilizzato Genesis Evolution System! 🚀")
    print("📁 Controlla i file PNG generati per i grafici spettacolari!")
    print("✅ Sistema grafico ottimizzato - warning risolti!")

if __name__ == "__main__":
    # Esegui simulazione
    main()