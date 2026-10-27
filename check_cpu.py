import os
import cpuinfo

# 1. Nombre de cœurs via le module standard
print(f"▶️ Nombre de cœurs (CPUs) : {os.cpu_count()}")

# 2. Récupération des technologies du processeur
info = cpuinfo.get_cpu_info()
flags = info.get("flags", [])

# 3. Vérification des instructions vectorielles
has_avx = "avx" in flags
has_avx2 = "avx2" in flags
has_fma = "fma" in flags

print(f"Modèle détecté : {info.get('brand_raw')}")
print(f"Support AVX  : {'OUI' if has_avx else 'NON'}")
print(f"Support AVX2 : {'OUI' if has_avx2 else 'NON'}")
print(f"Support FMA  : {'OUI' if has_fma else 'NON'}")
