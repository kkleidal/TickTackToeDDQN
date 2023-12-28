import numpy as np
import torch


class FFTickTackToeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(9, 32),
            torch.nn.GELU(),
            torch.nn.BatchNorm1d(32),
            torch.nn.Linear(32, 32),
            torch.nn.GELU(),
            torch.nn.BatchNorm1d(32),
            torch.nn.Linear(32, 9),
        )

    def forward(self, x):
        return self.layers(x).squeeze(1)

    def featurize(self, board, player):
        out = board.flatten().astype(np.float32)
        out[out == ((player + 1) % 2) + 1] = -1
        out[out == ((player) % 2) + 1] = 1
        return torch.from_numpy(out)


class Residual(torch.nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, x):
        return x + self.layer(x)


class SwapDims(torch.nn.Module):
    def __init__(self, dim1, dim2):
        super().__init__()
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, x):
        return x.transpose(self.dim1, self.dim2)


class TransformerTickTackToeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(6, 32),
            torch.nn.TransformerEncoderLayer(
                32, 4, dim_feedforward=64, batch_first=True
            ),
            torch.nn.TransformerEncoderLayer(
                32, 4, dim_feedforward=64, batch_first=True
            ),
            torch.nn.Linear(32, 1),
        )
        # self.layers = torch.nn.Sequential(
        #     torch.nn.Linear(9, 32),
        #     torch.nn.GELU(),
        #     torch.nn.BatchNorm1d(32),
        #     torch.nn.Linear(32, 32),
        #     torch.nn.GELU(),
        #     torch.nn.BatchNorm1d(32),
        #     torch.nn.Linear(32, 9),
        # )

    def forward(self, x):
        return self.layers(x).squeeze(-1)

    def featurize(self, board, player):
        return torch.from_numpy(
            np.stack(
                [
                    (board == player + 1).astype(np.float32),
                    (board == ((player + 1) % 2) + 1).astype(np.float32),
                    np.tile(np.expand_dims(np.arange(3), 0), [3, 1]).astype(np.float32),
                    np.tile(np.expand_dims(np.arange(3), 1), [1, 3]).astype(np.float32),
                    np.eye(3).astype(np.float32),
                    np.rot90(np.eye(3)).astype(np.float32),
                ],
                axis=-1,
            ).reshape([9, -1])
        )


# class TickTackToeModel(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.layers = torch.nn.Sequential(
#             torch.nn.Linear(9, 32),
#             torch.nn.GELU(),
#             torch.nn.BatchNorm1d(32),
#             torch.nn.Linear(32, 32),
#             torch.nn.GELU(),
#             torch.nn.BatchNorm1d(32),
#             torch.nn.Linear(32, 9),
#         )
#         # self.layers = torch.nn.Sequential(
#         #     torch.nn.Linear(15, 64),
#         #     torch.nn.ReLU(),
#         #     torch.nn.BatchNorm1d(64),
#         #     torch.nn.Linear(64, 8),
#         #     torch.nn.ReLU(),
#         #     torch.nn.BatchNorm1d(8),
#         #     torch.nn.Linear(8, 1),
#         # )
#         # self.layers = torch.nn.Sequential(
#         #     torch.nn.Conv2d(3, 16, 3, padding='same'),
#         #     torch.nn.ReLU(),
#         #     torch.nn.Conv2d(16, 1, 3, padding='same'),
#         # )
#         # self.layers = torch.nn.Sequential(
#         #     torch.nn.Conv2d(5, 16, 1),
#         #     TickTackToeBlock(16, 8),
#         #     # TickTackToeBlock(16, 8),
#         #     torch.nn.Conv2d(16, 1, 1),
#         # )

#     def forward(self, x):
#         return self.layers(x).squeeze(1)

#     def featurize(board, player):
#         out = board.flatten().astype(np.float32)
#         out[out == ((player+1)%2)+1] = -1
#         out[out == ((player)%2)+1] = 1
#         return torch.from_numpy(out)
#         # # row_ind = np.tile(np.expand_dims(np.arange(3), 1), [1, 3])
#         # # col_ind = np.tile(np.expand_dims(np.arange(3), 0), [3, 1])
#         # # parametrized = torch.from_numpy(np.stack([
#         # #     board == 0,
#         # #     board == (player+1),
#         # #     board == (((player+1)%2)+1),
#         # #     np.eye(3),  # is diag
#         # #     np.rot90(np.eye(3)),  # is anti-diag
#         # #     # *[
#         # #     #     row_ind == i
#         # #     #     for i in range(3)
#         # #     # ],
#         # #     # *[
#         # #     #     col_ind == i
#         # #     #     for i in range(3)
#         # #     # ],
#         # # ], axis=0).astype(np.float32))
#         # feats = []
#         # for i in range(3):
#         #     for j in range(3):
#         #         me = (player+1)
#         #         them = (((player+1)%2)+1)
#         #         same_row_self = np.count_nonzero(board[i] == me)
#         #         same_row_them = np.count_nonzero(board[i] == them)
#         #         same_col_self = np.count_nonzero(board[:, j] == me)
#         #         same_col_them = np.count_nonzero(board[:, j] == them)
#         #         empty_row = np.count_nonzero(board[i] == 0)
#         #         empty_col = np.count_nonzero(board[:, j] == 0)
#         #         diags = 0
#         #         diag = np.zeros([3, 3], dtype=bool)
#         #         if i == j:
#         #             diag |= np.eye(3, dtype=bool)
#         #             diags += 1
#         #         if i == 2 - j:
#         #             diag |= np.rot90(np.eye(3, dtype=bool))
#         #             diags += 1
#         #         same_diag_self = np.count_nonzero(board[diag] == me) if np.count_nonzero(diag) else 0
#         #         same_diag_them = np.count_nonzero(board[diag] == them) if np.count_nonzero(diag) else 0
#         #         empty_diag = np.count_nonzero(board[diag] == 0) if np.count_nonzero(diag) else 0
#         #         best_option_self = max(same_row_self, same_col_self, same_diag_self)
#         #         best_option_them = max(same_row_them, same_col_them, same_diag_them)
#         #         feats.append(np.array([
#         #             board[i, j] == 0,
#         #             board[i, j] == me,
#         #             board[i, j] == them,
#         #             best_option_self,
#         #             best_option_them,
#         #             empty_row,
#         #             empty_col,
#         #             empty_diag,
#         #             diags,
#         #         ], dtype=np.float32))
#         # feats = np.stack(feats, 1)
#         # feats = np.concatenate([feats, np.tile(np.max(feats[3:], axis=1, keepdims=True), [1, feats.shape[1]])], 0)

#         # feats = torch.from_numpy(feats)
#         # return feats
