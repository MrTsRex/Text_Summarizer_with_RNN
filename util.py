#File has utility functions
import time
import os
import tensorflow as tf
FG = tf.app.flags.FLAGS



def load_ckpt(saver, sess, ckpt_dir="train"): #loads checkpoint
  while True:
    try:

      ckpt_dir = os.path.join(FG.log_root, ckpt_dir)
      final_name_of_file = "checkpoint_best" if ckpt_dir=="eval" else None
      StateOfCheckpoint = tf.train.get_checkpoint_state(ckpt_dir, final_name_of_file=final_name_of_file)


      tf.logging.info('Load checkpoint %s', StateOfCheckpoint.model_checkpoint_path)
      saver.restore(sess, StateOfCheckpoint.model_checkpoint_path)
      return StateOfCheckpoint.model_checkpoint_path
    except:
      time.sleep(15)                #sleeps for 15 seconds
      tf.logging.info("There is a failure while loading checkpoint from %s. Sleeping for %i secs...", ckpt_dir, 15)

def get_config(): # will return configuartion
  
  Configuration = tf.ConfigProto(allow_soft_placement=True)
  Configuration.gpu_options.allow_growth=True
  return Configuration
      
