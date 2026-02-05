import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def calc_y(x: torch.Tensor) -> torch.Tensor:
    return 2.0 * x + 1.0 + 0.1

def main():
    device = get_device()
    logging.info(f"Using device: {device}")

    torch.manual_seed(0)

    N = 100
    x = torch.randn(N, 1)
    y = calc_y(x) + 0.1 * torch.randn(N, 1)

    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = nn.Linear(1, 1).to(device)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    model.train()
    for epoch in range(5):
        running_loss = 0.0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            preds = model(xb)
            loss = loss_fn(preds, yb)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        logging.info(f"epoch {epoch}: avg_loss={running_loss / len(loader):.4f}")

    #model.eval()
    with torch.no_grad():
        test_x = torch.tensor([[3.0]], device=device)
        test_y = model(test_x)
        logging.info(f"x=3.0 => y_hat={test_y.item():.4f}")
        logging.info(f"REFERENCE x=3.0 => y_hat={calc_y(3.0)}")


if __name__ == "__main__":
    main()
