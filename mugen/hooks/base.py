class BaseHook:
    def on_start(self):
        """
        Called when the training starts.
        """
    
    def on_end(self):
        """
        Called when the training ends.
        """
    
    def on_train_batch_start(self):
        """
        Called when the training starts for a batch.
        """
    
    def on_train_batch_end(self):
        """
        Called when the training ends for a batch.
        """
    
    def on_train_epoch_start(self):
        """
        Called when the training starts for an epoch.
        """
    
    def on_train_epoch_end(self):
        """
        Called when the training ends for an epoch.
        """
    
    def on_validation_epoch_start(self):
        """
        Called when the validation starts for an epoch.
        """

    def on_validation_epoch_end(self):
        """
        Called when the validation ends for an epoch.
        """
