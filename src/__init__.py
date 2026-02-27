# src is a package — required so that torch.save/load pickle paths of the
# form  src.CRNN_CTC.model.CRNN  resolve correctly when the repo root is
# on sys.path (which is the default when running from the project root).
