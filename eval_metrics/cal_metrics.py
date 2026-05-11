

def evaluate_and_save_batches(model, char2idx, idx2char, max_length,
                              training_smiles_set, out_csv_path,
                              num_batches=10, batch_size=64, temperature=1.0):
    all_generated = []
    for b in range(num_batches):
        generated = generate_smiles_batch(model, char2idx, idx2char,
                                          max_length, batch_size, temperature)
        all_generated.extend(generated)
        print(f"Batch {b+1}/{num_batches}: generated {len(generated)} SMILES")

    # Filtra e canonicalizza le SMILES valide
    valid_smiles = []
    for smi in all_generated:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            canon = Chem.MolToSmiles(mol, canonical=True)
            valid_smiles.append(canon)

    # Calcola novelty PRIMA di rimuovere duplicati
    if valid_smiles:
        num_novel = sum(1 for smi in valid_smiles if smi not in training_smiles_set)
        novelty_raw = num_novel / len(valid_smiles)
    else:
        novelty_raw = 0.0

    # Rimuovi duplicati e molecole già presenti nel training set
    unique_and_novel = list({smi for smi in valid_smiles if smi not in training_smiles_set})

    # Valuta QED e SA
    qed_list, sa_list = [], []
    for smi in unique_and_novel:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            try:
                qed_list.append(QED.qed(mol))
            except:
                pass
            try:
                sa_list.append(sascorer.calculateScore(mol))
            except:
                pass

    # Salva su CSV solo SMILES uniche e nuove
    with open(out_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        for smi in unique_and_novel:
            writer.writerow([smi])
    print(f"Saved {len(unique_and_novel)} unique and novel SMILES to {out_csv_path}")

    # Metriche
    validity = len(valid_smiles) / len(all_generated) if all_generated else 0
    avg_qed = np.mean(qed_list) if qed_list else 0
    avg_sa = np.mean(sa_list) if sa_list else 0
    originality = len(unique_and_novel) / len(valid_smiles) if valid_smiles else 0

    print(f"""
Molecule Generation Report:
  Total generated:         {len(all_generated)}
  Validity:               {validity*100:.2f}% ({len(valid_smiles)}/{len(all_generated)})
  Unique & novel:         {len(unique_and_novel)}
  Average QED:            {avg_qed:.4f}
  Average SA:             {avg_sa:.4f}
  Novelty (raw):          {novelty_raw*100:.2f}% 
  Originality (final):    {originality*100:.2f}% 
""")
