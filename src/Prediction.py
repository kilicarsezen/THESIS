import numpy as np

from Training import *

latent_samples = models['unconditioned_RealNVP'].distribution.sample(observations.shape[0]).numpy()
latent_samples_conditioned = models['conditioned_RealNVP'].distribution.sample(n_sample * observations.shape[0]).numpy()
energy_scores = []
predictions = []
for key, model in models.items():
    if key == 'unconditioned_RealNVP':
        prediction, _ = model.predict(latent_samples)
        predictions.append(prediction)
        energyscore_unc = np.empty(shape=(observations.shape[0]))
        for i in range(observations.shape[0]):
            energyscore_unc[i] = energy_score(observations.values[i], prediction)
        energyscore_avg = np.mean(energyscore_unc)
        energy_scores.append(energyscore_avg)

    if key == 'conditioned_RealNVP':
        latent_samples_conditioned = pd.concat([pd.DataFrame(latent_samples_conditioned,
                                                             columns=prices),
                                                external_factors_replicated.reset_index(drop=True)], axis=1)
        prediction, _ = model.predict_on_batch(latent_samples_conditioned)
        predictions.append(prediction)
        prediction_reshaped = np.reshape(prediction, (observations.shape[0], n_sample, len(prices)))
        energyscore = np.empty(shape=(observations.shape[0]))
        for i in range(observations.shape[0]):
            energyscore[i] = energy_score(observations.values[i], prediction_reshaped[i, :, :])
        energy_scores.append(np.mean(energyscore))

    if key == 'conditioned_benchmark':
        prediction = model.predict(external_factors_replicated.reset_index(drop=True), batch_size=1)
        predictions.append(prediction)
        Y_pred_cond_reshaped = np.reshape(prediction, (observations.shape[0], n_sample, len(prices)))
        energyscore_bench_cond = np.empty(shape=(observations.shape[0]))
        for i in range(observations.shape[0]):
             energyscore_bench_cond[i] = energy_score(observations.values[i], Y_pred_cond_reshaped[i, :, :])
        energy_scores.append(np.mean(energyscore_bench_cond))
    if key == 'copula_benchmark':
        prediction = []
        for i in observations.shape[0]:
            prediction = model.sample(10,external_factors[i])
            prediction.append(prediction)
        predictions.append(prediction)

# unconditional benchmark
Y_pred_unc = unc_bench.sample(observations.shape[0]).numpy()
predictions.append(Y_pred_unc)
energyscore_unc_benc = np.empty(shape=(observations.shape[0]))
for i in range(observations.shape[0]):
    energyscore_unc_benc[i] = energy_score(observations.values[i], Y_pred_unc)
energy_scores.append(np.mean(energyscore_unc_benc))
