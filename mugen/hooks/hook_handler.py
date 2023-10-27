from typing import List
from .base import BaseHook


class HookHandler:
    def __init__(self):
        self._hooks: List[BaseHook] = []
    
    def register_hook(self, hook: BaseHook):
        self._hooks.append(hook)

    def remove_hook(self, hook: BaseHook):
        self._hooks.remove(hook)
    
    def on_start(self):
        for hook in self._hooks:
            hook.on_start()
    
    def on_end(self):
        for hook in self._hooks:
            hook.on_end()
    
    def on_train_batch_start(self):
        for hook in self._hooks:
            hook.on_train_batch_start()
    
    def on_train_batch_end(self):
        for hook in self._hooks:
            hook.on_train_batch_end()
    
    def on_train_epoch_start(self):
        for hook in self._hooks:
            hook.on_train_epoch_start()
    
    def on_train_epoch_end(self):
        for hook in self._hooks:
            hook.on_train_epoch_end()
    
    def on_validation_epoch_start(self):
        for hook in self._hooks:
            hook.on_validation_epoch_start()
    
    def on_validation_epoch_end(self):
        for hook in self._hooks:
            hook.on_validation_epoch_end()
