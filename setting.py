import numpy as np 
from scipy.spatial import distance

# -----------------------------
# Selected wells (seed2026)
# -----------------------------
SEED = 2026
SELECTED_WELLS_CSV = r"data/Stanford_VI/selected_wells_20_seed2026.csv"

TCN1D_train_p = {'batch_size': 4,
				'no_wells': 401, 
				'unsupervised_seismic': -1,
                'lambda_ai': 5.0,
                'lambda_facies': 0.2,
                'lambda_recon': 1.0,
                'lambda_recon_ws': 0.2,
                'facies_detach_y': False,
				'SGS_data_n': 1000,
				'epochs': 1000, 
				'grad_clip': 1.0,
				'input_shape': (1, 200),
				'lr': 0.0001,
                # DataLoader acceleration
				'num_workers': 4,             # Windows寤鸿 2~8锛涘厛鐢?
				'pin_memory': True,           # GPU璁粌寤鸿True
				'persistent_workers': False,   # num_workers>0 鏃舵湁鏁?                # Weak-supervision scheduling 
				'deterministic': True,
				'use_deterministic_algorithms': True,
				'deterministic_warn_only': True,
				'cudnn_deterministic': True,
				'cudnn_benchmark': False,
				'set_cublas_workspace_config': True,
				'cublas_workspace_config': ':4096:8',
				'set_pythonhashseed': True,
				'disable_tf32': True,
				'loader_num_workers': 0,
				'log_reproducibility': True,
				'seeds': '',
				'ws_every': 2,                # 姣?涓猠poch璺戜竴娆″急鐩戠潱锛堝ぇ骞呮彁閫燂級
                'ws_max_batches': 150,         # 姣忔鏈€澶氳窇50涓急鐩戠潱batch锛堝啀鎻愰€燂級
                'ws_warmup_epochs': 100,
                "seed": SEED,
                "no_wells": 20,                   # 鉁?鐢ㄧ湡瀹?20 鍙ｄ簳
                "selected_wells_csv": SELECTED_WELLS_CSV,   # 鉁?浜曚綅绉嶅瓙鏉ユ簮

				# -----------------------------
				# Facies-adaptive anisotropic conditioning (FARP)
				# A reliability field R(x) is propagated from sparse wells in a facies- and
				# waveform-similarity induced anisotropic metric space, then injected as an
				# extra input channel (and optional soft prior loss).
				# NOTE: default is False to keep original GPI behavior.
				'use_aniso_conditioning': True,
				'channel_id': 2,            # VI-E: 2 is Channel
				'aniso_steps_R': 25,
				'aniso_eta': 0.6,
				'aniso_gamma': 8.0,
				'aniso_tau': 0.6,
				'aniso_kappa': 4.0,
				'aniso_sigma_st': 1.2,
				# noise- & uncertainty-aware adaptive anisotropic evolution (proposed_v2)
				'adaptive_eta_enable': False,
				'eta_min': 0.0,
				'eta_floor_ratio': 0.20,
				'alpha_floor_ratio': 0.20,
				'snr_low': 10.0,
				'snr_high': 30.0,
				'snr_power': 2.0,
				'ent_power': 0.5,
				'phys_power': 0.5,
				'tau': 0.60,
				's': 0.08,
				'k': 1.0,
				'enable_margin_gate': True,
				'margin0': 0.15,
				'margin_s': 0.05,
				'w_ent': 0.5,
				'w_phys': 0.5,
				'alpha_update_mode': 'decoupled',
				'eta_update_mode': 'decoupled',
				'enable_suite_final5': False,
				'rho_skip': 0.10,
				'snr_freeze': 12.0,
				'snr_cap_by_residual': True,
				'rho_warmup_ratio': 0.,
				'gamma_warmup_ratio': 0.30,
				'gamma_cap_ratio': 0.80,
				'adaptive_eps': 1e-8,
				'residual_ema_decay': 0.9,
				'residual_pctl': 95.0,
				'eta_log_every': 1,
				'eta_warn_ratio_thr': 0.1,
				# optional soft impedance prior (propagated from wells)
				'use_soft_prior': False,
				'aniso_steps_prior': 35,
				'lambda_prior': 0.20,
				# -----------------------------
				# Iterative coupling for R(x) (optional)
				# Rebuild R from predicted facies + physics residual every few epochs,
				# and EMA-update to avoid drift.
				'iterative_R': True,
				'R_update_every': 50,
				'R_update_bs': 16,
				'R_ema_beta': 0.85,
				'alpha_prior_start': 1.0,
				'alpha_prior_end': 0.3,
				'alpha_prior_decay_epochs': 800,
				'conf_thresh': 0.75,
				'lambda_phys_damp': 0.8,
				'save_R_every': 50,


    # SOFT妯″瀷锛?銆乂ishalNet, 2銆丟RU_MM, 3銆乁net_1D, 4銆乁net_1D_convolution
    # 鍙嶆紨妯″瀷锛歍ransformer_cov_para_geo锛歍ransformer+cov_para+geo
    # 娑堣瀺瀹為獙妯″瀷锛?1銆乀ansformer_cov_para, 2銆乀ansformer_geo, 3銆乀ansformer_convolution_geo

				'model_name': 'VishalNet',
				'Forward_model': 'cov_para',       # '' 琛ㄧず娌℃湁姝ｆ紨杩囩▼
				'Facies_model': 'Facies',

	# 'Stanford_VI', 'Fanny', 
				'data_flag': 'Stanford_VI',
				'get_F': 0,  #锛?,2,4锛?鍦伴渿鏁版嵁鎵╁厖浜嗛鐜囩壒寰佸拰鍔ㄦ€佺壒寰侊紝褰?鏃惰〃绀哄彧鏈夋椂鍩熷湴闇囨尝褰?				'F': 'WE_PreSDM',  # 褰擄細"data_flag = M2_F
				}

TCN1D_test_p = {"no_wells": 20,                   # 鉁?鐢ㄧ湡瀹?20 鍙ｄ簳
                "seed": SEED,
                "selected_wells_csv": SELECTED_WELLS_CSV,   # 鉁?浜曚綅绉嶅瓙鏉ユ簮               
				'data_flag':'Stanford_VI',
				'model_name': 'VishalNet_cov_para_Facies_s_uns',  # model-name_Forward-model_Facies-model_s_uns
                "run_id": "VishalNet_cov_para_Facies",   # 鉁?鐢ㄤ簬鎵?full_ckpt/stats
				# 娉ㄦ剰锛屽鏋滄槸鏈夋婕旀ā鍧楋紝鍒欙細鍙嶆紨妯″潡_姝ｆ紨妯″潡
				# If you trained with anisotropic conditioning, enable it in test as well.
				'use_aniso_conditioning': True,
				}
