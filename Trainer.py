import torch
import time
import os
import Config

device = Config.DEVICE
model_path = Config.MODEL_PATH

##################################################################
# This is a plain pytorch training loop with validation embedded
# But the problem is also clear, no logging for loss, no visulaization for the loss/accuracy
# to improve it, we can use ignite
def train_classification_model(data, model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    dataloaders, class_names, dataset_sizes = data
    print("class names are: ", class_names)
    best_model_params_path = os.path.join(model_path, 'best_model_params.pt')

    torch.save(model.state_dict(), best_model_params_path)
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), best_model_params_path)

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(torch.load(best_model_params_path))
    return model
##################################################################


##################################################################
# This uses pytorch ignite + tensorboard logger 
# 
# https://pytorch.org/ignite/generated/ignite.handlers.param_scheduler.LRScheduler.html#ignite.handlers.param_scheduler.LRScheduler
from ignite.engine import Engine, Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from ignite.handlers import ModelCheckpoint
from ignite.handlers.param_scheduler import LRScheduler
from ignite.contrib.handlers import TensorboardLogger, global_step_from_engine

def get_trainer(dataloaders, model, optimizer, criterion, log_interval = 100):

    train_loader = dataloaders['train']
    val_loader = dataloaders['val']
    trainer = create_supervised_trainer(model, optimizer, criterion, device)
    val_metrics = {
        "accuracy": Accuracy(),
        "loss": Loss(criterion)
    }

    train_evaluator = create_supervised_evaluator(model, metrics=val_metrics, device=device)
    val_evaluator = create_supervised_evaluator(model, metrics=val_metrics, device=device)
    # How many batches to wait before logging training status

    def log_training_loss(engine):
        print(f"Epoch[{engine.state.epoch}], Iter[{engine.state.iteration}] Loss: {engine.state.output:.2f}")
    trainer.add_event_handler(Events.ITERATION_COMPLETED(every=log_interval), log_training_loss)

    def log_training_results(trainer):
        train_evaluator.run(train_loader)
        metrics = train_evaluator.state.metrics
        print(f"Training Results - Epoch[{trainer.state.epoch}] Avg accuracy: {metrics['accuracy']:.2f} Avg loss: {metrics['loss']:.2f}")
    trainer.add_event_handler(Events.EPOCH_COMPLETED, log_training_results)

    def log_validation_results(trainer):
        val_evaluator.run(val_loader)
        metrics = val_evaluator.state.metrics
        print(f"Validation Results - Epoch[{trainer.state.epoch}] Avg accuracy: {metrics['accuracy']:.2f} Avg loss: {metrics['loss']:.2f}")
    trainer.add_event_handler(Events.EPOCH_COMPLETED, log_validation_results)

    def score_function(engine):
        return engine.state.metrics["accuracy"]

    model_checkpoint_path = Config.MODEL_PATH
    # Checkpoint to store n_saved best models wrt score function
    model_checkpoint = ModelCheckpoint(
        model_checkpoint_path, 
        n_saved=2,
        filename_prefix="best",
        require_empty=False,
        score_function=score_function,
        score_name="accuracy",
        global_step_transform=global_step_from_engine(trainer), # helps fetch the trainer's state
    )
    
    # Save the model after every epoch of val_evaluator is completed
    val_evaluator.add_event_handler(Events.COMPLETED, model_checkpoint, {"model": model})

    # Define a Tensorboard logger
    tb_logger = TensorboardLogger(log_dir="tb-logger")

    # Attach handler to plot trainer's loss every 100 iterations
    tb_logger.attach_output_handler(
        trainer,
        event_name=Events.ITERATION_COMPLETED(every=log_interval),
        tag="training",
        output_transform=lambda loss: {"batch_loss": loss},
    )

    # Attach handler for plotting both evaluators' metrics after every epoch completes
    for tag, evaluator in [("training", train_evaluator), ("validation", val_evaluator)]:
        tb_logger.attach_output_handler(
            evaluator,
            event_name=Events.EPOCH_COMPLETED,
            tag=tag,
            metric_names="all",
            global_step_transform=global_step_from_engine(trainer),
        )
    return trainer

def train_classification_model_ignite(data, model, criterion, optimizer, scheduler = None, num_epochs = 5):
    dataloaders, class_names, dataset_sizes = data
    print("class names are: ", class_names)
    trainer = get_trainer(dataloaders, model, optimizer, criterion)
    if scheduler is not None:
        scheduler = LRScheduler(scheduler)
        def print_lr():
            print(f"learning rate: {optimizer.param_groups[0]['lr']}")
        trainer.add_event_handler(Events.EPOCH_COMPLETED, print_lr)
    trainer.run(dataloaders['train'], max_epochs=num_epochs)

##################################################################
