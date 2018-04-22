
"""This file contains code to build and run the tensorflow graph for the sequence-to-sequence model"""
import time
import os
import tensorflow as tf
import numpy as np
from tensorflow.contrib.tensorboard.plugins import projector
from attention_decoder import attention_decoder

FLAGS = tf.app.flags.FLAGS

class SummarizationModel(object):
  """It shows text summarization"""

  def __init__(self, hps, vocab):
    self._hps = hps
    self._vocab = vocab

  def _Placeholder_adding(self):
    """Adding placeholders"""
    hps = self._hps
 
    if  hps.coverage and hps.mode=="decode":   #decoder
      self.prev_coverage = tf.placeholder(tf.float32, [hps.batch_size, None], name='prev_coverage')
   
    self._DecoderMask = tf.placeholder(tf.float32, [hps.batch_size, hps.max_dec_steps], name='DecoderMask')
    self._B_Decoded = tf.placeholder(tf.int32, [hps.batch_size, hps.max_dec_steps], name='B_Decoded')
    self._Tar = tf.placeholder(tf.int32, [hps.batch_size, hps.max_dec_steps], name='Tar')

    
    
    self._EncoderMask = tf.placeholder(tf.float32, [hps.batch_size, None], name='EncoderMask')   #encoder
    self._BEncoded = tf.placeholder(tf.int32, [hps.batch_size, None], name='BEncoded')
    self._LensEncoded = tf.placeholder(tf.int32, [hps.batch_size], name='LensEncoded')
    if FLAGS.pointer_gen: 
      self._OOVSMaximum = tf.placeholder(tf.int32, [], name='OOVSMaximum')   
      self._BEncoded_extend_vocab = tf.placeholder(tf.int32, [hps.batch_size, None], name='BEncoded_extend_vocab') 


  

  def _EncoderFn(self, encoder_inputs, seq_len):
   #This implements single layer bidirectional LSTM
    with tf.variable_scope('encoder'):
      
      Backward = tf.contrib.rnn.GRUCell(self._hps.hidden_dim, initializer=self.rand_unif_init, state_is_tuple=True)
      Forward= tf.contrib.rnn.GRUCell(self._hps.hidden_dim, initializer=self.rand_unif_init, state_is_tuple=True)
      (EncOut, (forwardState, backwardState)) = tf.nn.bidirectional_dynamic_rnn(Forward, Backward, encoder_inputs, dtype=tf.float32, sequence_length=seq_len, swap_memory=True)
      EncOut = tf.concat(axis=2, values=EncOut) 
    return EncOut, forwardState, backwardState


  def _Reduction(self, forwardState, backwardState):
    #Reducing Forward and backward to single graph
    hidden_dim = self._hps.hidden_dim
    with tf.variable_scope('reduce_final_st'):

  def _FD_Create(self, batch, just_enc=False):
    #mapping
    FD = {}
    FD[self._LensEncoded] = batch.LensEncoded
    FD[self._EncoderMask] = batch.EncoderMask
    FD[self._BEncoded] = batch.BEncoded
    
    
    if not just_enc:
      FD[self._B_Decoded] = batch.B_Decoded
      FD[self._Tar] = batch.Tar
      FD[self._DecoderMask] = batch.DecoderMask

    if FLAGS.pointer_gen:
      FD[self._BEncoded_extend_vocab] = batch.BEncoded_extend_vocab
      FD[self._OOVSMaximum] = batch.OOVSMaximum
    return FD    

      #Weights and biases
      hbias = tf.get_variable('hbias', [hidden_dim], dtype=tf.float32, initializer=self.trunc_norm_init)
      cbias = tf.get_variable('cbias', [hidden_dim], dtype=tf.float32, initializer=self.trunc_norm_init)

      HWeightRed = tf.get_variable('HWeightRed', [hidden_dim * 2, hidden_dim], dtype=tf.float32, initializer=self.trunc_norm_init)
      CWeightReduce = tf.get_variable('CWeightReduce', [hidden_dim * 2, hidden_dim], dtype=tf.float32, initializer=self.trunc_norm_init)
      
      # Concatenating fw and bw cell, also concatenatinating fw and bw state
      c_prev = tf.concat(axis=1, values=[forwardState.c, backwardState.c]) 
      h_prev = tf.concat(axis=1, values=[forwardState.h, backwardState.h]) 
      
      #obtaining new state from old state, obtaining new cell from old cell
      h_modified = tf.nn.relu(tf.matmul(h_prev, HWeightRed) + hbias) 
      c_modified = tf.nn.relu(tf.matmul(c_prev, CWeightReduce) + cbias) 
      return tf.contrib.rnn.LSTMStateTuple(c_modified, h_modified)


  

  def _FinalCalculation(self, vocab_dists, DistributionAttention):
    #pointer-generator
    with tf.variable_scope('final_distribution'):
      
     
      DistributionAttention = [(1-p_gen) * dist for (p_gen,dist) in zip(self.ppp, DistributionAttention)]
      vocab_dists = [p_gen * dist for (p_gen,dist) in zip(self.ppp, vocab_dists)]

      
      VSzExt = self._vocab.size() + self._OOVSMaximum # extended vocabulary's max size
      ZZero = tf.zeros((self._hps.batch_size, self._OOVSMaximum))
      ExtVocabularyDistribution = [tf.concat(axis=1, values=[dist, ZZero]) for dist in vocab_dists] 
      
      number_of_batches = tf.range(0, limit=self._hps.batch_size) 
      number_of_batches = tf.expand_dims(number_of_batches, 1) 
      length = tf.shape(self._BEncoded_extend_vocab)[1] # no. of states
      number_of_batches = tf.tile(number_of_batches, [1, length]) 
      shape = [self._hps.batch_size, VSzExt]
      indd = tf.stack( (number_of_batches, self._BEncoded_extend_vocab), axis=2) # converts shape to (batch_size, enc_t, 2)
      ProjDistAttention = [tf.scatter_nd(indd, copy_dist, shape) for copy_dist in DistributionAttention] 

      
      DistributionFInal = [vocab_dist + copy_dist for (vocab_dist,copy_dist) in zip(ExtVocabularyDistribution, ProjDistAttention)]


      return DistributionFInal


def _DEcoderFn(self, inputs):
    #Attention decoder function
    hps = self._hps
    cell = tf.contrib.rnn.GRUCell(hps.hidden_dim, state_is_tuple=True, initializer=self.rand_unif_init)

    prev_coverage = self.prev_coverage if hps.mode=="decode" and hps.coverage else None 

    resss, StateOut, DistributionAttention, ppp, coverage = attention_decoder(inputs, self._dec_in_state, self._EnccState, self._EncoderMask, cell, initial_state_attention=(hps.mode=="decode"), pointer_gen=hps.pointer_gen, use_coverage=hps.coverage, prev_coverage=prev_coverage)

    return resss, StateOut, DistributionAttention, ppp, coverage


  
  def _SeqtoSEq(self):
    #adding sequence to sequence model
    hps = self._hps
    VocSz = self._vocab.size() # vocabulary's size

    with tf.variable_scope('seq2seq'):
      
      
      self.trunc_norm_init = tf.truncated_normal_initializer(stddev=hps.trunc_norm_init_std)
      self.rand_unif_init = tf.random_uniform_initializer(-hps.rand_unif_init_mag, hps.rand_unif_init_mag, seed=123)

     
      with tf.variable_scope('embedding'): #adding embedding matrix
        embedding = tf.get_variable('embedding', [VocSz, hps.emb_dim], dtype=tf.float32, initializer=self.trunc_norm_init)

        if hps.mode=="train": self._EmbeddedFn(embedding) 
        DecoderInput_Embedded = [tf.nn.embedding_lookup(embedding, x) for x in tf.unstack(self._B_Decoded, axis=1)] 
        EncoderInputs_Embedded = tf.nn.embedding_lookup(embedding, self._BEncoded) 

      
      self._EnccState = enc_outputs #encoder is added
      enc_outputs, forwardState, backwardState = self._EncoderFn(EncoderInputs_Embedded, self._LensEncoded)

      
      self._dec_in_state = self._Reduction(forwardState, backwardState)

      
      with tf.variable_scope('decoder'):  #adding decoder
        decoder_outputs, self._dec_out_state, self.DistributionAttention, self.ppp, self.coverage = self._DEcoderFn(DecoderInput_Embedded)

      
      with tf.variable_scope('output_projection'):
        Score = [] 
        vv = tf.get_variable('v', [VocSz], dtype=tf.float32, initializer=self.trunc_norm_init)
        
        ww = tf.get_variable('ww', [hps.hidden_dim, VocSz], dtype=tf.float32, initializer=self.trunc_norm_init)
        ww_tt = tf.transpose(ww)
        
        for ii,output in enumerate(decoder_outputs):
          if ii > 0:
            tf.get_variable_scope().reuse_variables()
          Score.append(tf.nn.xw_plus_b(output, ww, vv)) # apply the linear layer

        vocab_dists = [tf.nn.softmax(g) for g in Score] 

      
      if not FLAGS.pointer_gen:
        DistributionFInal = vocab_dists
        
      else: 
        DistributionFInal = self._FinalCalculation(vocab_dists, self.DistributionAttention)



      if hps.mode in ['train', 'eval']:
        # Loss calculation
        with tf.variable_scope('loss'):
          if FLAGS.pointer_gen:
            number_of_batches = tf.range(0, limit=hps.batch_size) 
            EachStepLoss = [] 
            
            for dec_step, dist in enumerate(DistributionFInal):

              Tars = self._Tar[:,dec_step] 
              indd = tf.stack( (number_of_batches, Tars), axis=1) 
              g_b = tf.gather_nd(dist, indd) 
              losses = -tf.log(g_b)
              EachStepLoss.append(losses)

            
            self._loss = _Average_And_Mask(EachStepLoss, self._DecoderMask)

          else: 
            self._loss = tf.contrib.seq2seq.sequence_loss(tf.stack(Score, axis=1), self._Tar, self._DecoderMask) # baseline model

          tf.summary.scalar('loss', self._loss)

          
          if hps.coverage: #calculating coverage loss
            with tf.variable_scope('coverage_loss'):
              self._CalculatingCoverage = _CalculatingCoverage(self.DistributionAttention, self._DecoderMask)

              tf.summary.scalar('coverage_loss', self._CalculatingCoverage)
            self._total_loss = self._loss + hps.cov_loss_wt * self._CalculatingCoverage

            tf.summary.scalar('total_loss', self._total_loss)

    if hps.mode == "decode":
      
      assert len(DistributionFInal)==1 
      DistributionFInal = DistributionFInal[0]
      topk_probs, self._topk_ids = tf.nn.top_k(DistributionFInal, hps.batch_size*2) 
  
      self._topk_log_probs = tf.log(topk_probs)



  def _Train_Addition(self):
    
    
    tvars = tf.trainable_variables()
    loss_to_minimize = self._total_loss if self._hps.coverage else self._loss
    gradients = tf.gradients(loss_to_minimize, tvars, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)

   
    with tf.device("/gpu:0"):  #gradient clipping
      grads, global_norm = tf.clip_by_global_norm(gradients, self._hps.max_grad_norm)

    
    tf.summary.scalar('global_norm', global_norm)  #adding summary

    
    Opt = tf.train.AdagradOptimizer(self._hps.lr, initial_accumulator_value=self._hps.adagrad_init_acc) #applying adagrad opt
    with tf.device("/gpu:0"):
      self._train_op = Opt.apply_gradients(zip(grads, tvars), global_step=self.global_step, name='train_step')

  def _EmbeddedFn(self, embedding_var):
   
    D_Train = os.path.join(FLAGS.log_root, "train")
    PathOfMetadataPath = os.path.join(D_Train, "vocab_metadata.tsv")
    self._vocab.write_metadata(PathOfMetadataPath)
    configurations_new = projector.ProjectorConfig()
    Summ = tf.summary.FileWriter(D_Train)



    embedding = configurations_new.embeddings.add()
    embedding.tensor_name = embedding_var.name
    embedding.metadata_path = PathOfMetadataPath
    projector.visualize_embeddings(summary_writer, configurations_new)

  
  def Training_Run(self, sess, batch):
    #Running only one iteration for training
    FD = self._FD_Create(batch)
    to_return = {
        'summaries': self._summaries,
        'train_op': self._train_op,
        'global_step': self.global_step,
        'loss': self._loss,
    }
    if self._hps.coverage:
      to_return['coverage_loss'] = self._CalculatingCoverage
    return sess.run(to_return, FD)

  def Running_Evaluation(self, sess, batch):
    #running one iteration for evaluation
    FD = self._FD_Create(batch)
    to_return = {
      'loss': self._loss,
      'global_step': self.global_step,
      'summaries': self._summaries,
        
    }

    if self._hps.coverage:
      to_return['coverage_loss'] = self._CalculatingCoverage
    return sess.run(to_return, FD)

  def Building_GRaph(self):
    
    tf.logging.info('Building graph...')
    tinn = time.time()
    self._Placeholder_adding()
    
    with tf.device("/gpu:0"):
      self._SeqtoSEq()
    self.global_step = tf.Variable(0, name='global_step', trainable=False)
    if self._hps.mode == 'train':
      self._Train_Addition()
    self._summaries = tf.summary.merge_all()
    tout = time.time()
    tf.logging.info('Time to build graph: %i seconds', tout - tinn)

  def In_OneStep_Decode(self, sess, batch, latest_tokens, EnccState, dec_init_states, prev_coverage):
    #this is for beam search decoding

    beam_size = len(dec_init_states)

    
    hiddens = [np.expand_dims(state.h, axis=0) for state in dec_init_states] #list of LSTMStateTuples is converted to single LSTMTuple
    cells = [np.expand_dims(state.c, axis=0) for state in dec_init_states]
    h_modified = np.concatenate(hiddens, axis=0)  
    c_modified = np.concatenate(cells, axis=0)  
    
    new_dec_in_state = tf.contrib.rnn.LSTMStateTuple(c_modified, h_modified)

    feed = {
        self._dec_in_state: new_dec_in_state,
        self._B_Decoded: np.transpose(np.array([latest_tokens])),
        self._EncoderMask: batch.EncoderMask,
        self._EnccState: EnccState,
    }

    to_return = {
      "probs": self._topk_log_probs,
      "states": self._dec_out_state,
      "ids": self._topk_ids,
      "DistributionAttention": self.DistributionAttention
    }

    

    if self._hps.coverage:
      feed[self.prev_coverage] = np.stack(prev_coverage, axis=0)
      to_return['coverage'] = self.coverage

    if FLAGS.pointer_gen:
      feed[self._BEncoded_extend_vocab] = batch.BEncoded_extend_vocab
      feed[self._OOVSMaximum] = batch.OOVSMaximum
      to_return['ppp'] = self.ppp  

    results = sess.run(to_return, FD=feed) # run the decoder step

    
    new_states = [tf.contrib.rnn.LSTMStateTuple(results['states'].c[i, :], results['states'].h[i, :]) for i in xrange(beam_size)]

   
    assert len(results['DistributionAttention'])==1
    DistributionAttention = results['DistributionAttention'][0].tolist()

    if FLAGS.pointer_gen: #Converting singleton list to list of k arrays
      
      assert len(results['ppp'])==1
      ppp = results['ppp'][0].tolist()
    else:
      ppp = [None for _ in xrange(beam_size)]

    #this is done for each hypothesis
    if FLAGS.coverage:
      CovgNew = results['coverage'].tolist()
      assert len(CovgNew) == beam_size
    else:
      CovgNew = [None for _ in xrange(beam_size)]

    return results['ids'], results['probs'], new_states, DistributionAttention, ppp, CovgNew
  

  def EncoderRunning(self, sess, batch):
    #this function returns encoder states and decoder states
    FD = self._FD_Create(batch, just_enc=True) 
    dec_in_state = tf.contrib.rnn.LSTMStateTuple(dec_in_state.c[0], dec_in_state.h[0])
    (EnccState, dec_in_state, global_step) = sess.run([self._EnccState, self._dec_in_state, self.global_step], FD) # run the encoder

    
    return EnccState, dec_in_state


  




def _CalculatingCoverage(DistributionAttention, padding_mask):
  """Calculates the coverage loss from the attention distributions.
  """
  covlosses = [] 
  coverage = tf.zeros_like(DistributionAttention[0]) 
  
  for cc in DistributionAttention:
    covloss = tf.reduce_sum(tf.minimum(cc, coverage), [1]) 
    covlosses.append(covloss)
    coverage += cc 
  coverage_loss = _Average_And_Mask(covlosses, padding_mask)
  return coverage_loss

def _Average_And_Mask(values, padding_mask):
  """Applies mask to values then returns overall average (a scalar)

 
  """
  ValEachStep = [v * padding_mask[:,dec_step] for dec_step,v in enumerate(values)]
  DecoderLens = tf.reduce_sum(padding_mask, axis=1)
  values_per_ex = sum(ValEachStep)/DecoderLens 
  return tf.reduce_mean(values_per_ex) 