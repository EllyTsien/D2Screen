def generate_cmpx_pdb(inpt_prot: str, inpt_pose: str, oupt_cmpx: str) -> None:

    def write_line(line: str, keywords: list, oupt_file: str) -> None:
        header = line.split()[0]
        if header in keywords:
            oupt_file.write(line)

    def cmpx_writer() -> None:
        with open(oupt_cmpx, 'w') as oupt_file, \
             open(inpt_prot, 'r') as prot_file, \
             open(inpt_pose, 'r') as pose_file:
            for prot_line in prot_file:
                write_line(prot_line, prot_headers, oupt_file)
            for pose_line in pose_file:
                write_line(pose_line, pose_headers, oupt_file)

    prot_headers = ['ATOM', 'CONECT', 'TER']
    pose_headers = ['ATOM', 'CONECT', 'END']
    cmpx_writer()

count = 0
for folder in tqdm(LIG_dFLDs):
    LIG_ID = os.path.basename(folder)
    LIG_iFLD = os.path.join(INT_FLD, LIG_ID)
    os.makedirs(LIG_iFLD, exist_ok=True)

    LIG_pdb_dFFiles = sorted(glob.glob(folder + '/*.pdb'))
    for pose_file in LIG_pdb_dFFiles:
        pose_name = os.path.basename(pose_file).split('.')[0]
        cmpx_pdb = pose_name + '_cmpx.pdb'
        cmpx_pdb_ifile = os.path.join(LIG_iFLD, cmpx_pdb)
        generate_cmpx_pdb(PROT_pdb_dFile, pose_file, cmpx_pdb_ifile)
        count += 1

print(f'+ {count} ligand_[n]_cmpx.pdb > INTERACTION folder')