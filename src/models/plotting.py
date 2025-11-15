import matplotlib.pyplot as plt

def plot_loss_curve(trainer):
  # plot the loss curve
  train_losses = [x["loss"] for x in trainer.state.log_history if "loss" in x]
  
  plt.figure(figsize=(8,5))
  plt.plot(train_losses, label="Train Loss")
  plt.xlabel("Steps")
  plt.ylabel("Loss")
  plt.title("Training Loss")
  plt.legend()
  plt.show()
