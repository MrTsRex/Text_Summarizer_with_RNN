"""This file contains code to build and run the tensorflow graph for the sequence-to-sequence model"""
import time
import os
import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
from attention_decoder import attention_decoder

FLAGS = tf.app.flags.FLAGS

class SummarizationModel(object):
  """A class to represent a sequence-to-sequence model for text summarization. Supports both baseline mode, pointer-generator mode, and coverage"""

  def __init__(self, hps, vocab):
    
    self._vocab = vocab
    self._hps = hps

  def _add_placeholders(self):
    """Add placeholders to the graph. These are entry points for any input data."""
    hps = self._hps

    # encoder part
    
    self._EncoderMask = tf.placeholder(tf.float32, [hps.batch_size, None], name='enc_padding_mask')
    self._LensEncoder = tf.placeholder(tf.int32, [hps.batch_size], name='enc_lens')
    self._BatchEncoder = tf.placeholder(tf.int32, [hps.batch_size, None], name='enc_batch')
    
    if FLAGS.pointer_gen:
       self._MaximumOOvs = tf.placeholder(tf.int32, [], name='max_art_oovs')
       self._BatchEncoder_extend_vocab = tf.placeholder(tf.int32, [hps.batch_size, None], name='enc_batch_extend_vocab')
    

    # decoder part
    self._BDecoder = tf.placeholder(tf.int32, [hps.batch_size, hps.max_dec_steps], name='dec_batch')
    self._MaskDecoderPad = tf.placeholder(tf.float32, [hps.batch_size, hps.max_dec_steps], name='dec_padding_mask')
    self._BatchTar = tf.placeholder(tf.int32, [hps.batch_size, hps.max_dec_steps], name='target_batch')

    if hps.mode=="decode" and hps.coverage:
      self.prev_coverage = tf.placeholder(tf.float32, [hps.batch_size, None], name='prev_coverage')


  def _make_feed_dict(self, batch, just_enc=False):
    """Make a feed dictionary mapping parts of the batch to the appropriate placeholders.

    Args:
      batch: Batch object
      just_enc: Boolean. If True, only feed the parts needed for the encoder.
    """
    FD = {}
    FD[self._LensEncoder] = batch.enc_lens
    FD[self._EncoderMask] = batch.enc_padding_mask
    FD[self._BatchEncoder] = batch.enc_batch
    if FLAGS.pointer_gen:
      FD[self._BatchEncoder_extend_vocab] = batch.enc_batch_extend_vocab
      FD[self._MaximumOOvs] = batch.max_art_oovs
    if not just_enc:
      FD[self._MaskDecoderPad] = batch.dec_padding_mask
      FD[self._BatchTar] = batch.target_batch
      FD[self._BDecoder] = batch.dec_batch
    return FD

  def _add_encoder(self, encoder_inputs, seq_len):
    """Add a single-layer bidirectional LSTM encoder to the graph.

    Args:
      encoder_inputs: A tensor of shape [batch_size, <=max_enc_steps, emb_size].
      seq_len: Lengths of encoder_inputs (before padding). A tensor of shape [batch_size].

    Returns:
      OutEnc:
        A tensor of shape [batch_size, <=max_enc_steps, 2*hidden_dim]. It's 2*hidden_dim because it's the concatenation of the forwards and backwards states.
      fw_state, bw_state:
        Each are LSTMStateTuples of shape ([batch_size,hidden_dim],[batch_size,hidden_dim])
    """
    with tf.variable_scope('encoder'):
      back = tf.contrib.rnn.LSTMCell(self._hps.hidden_dim, initializer=self.rand_unif_init, state_is_tuple=True)
      Forward = tf.contrib.rnn.LSTMCell(self._hps.hidden_dim, initializer=self.rand_unif_init, state_is_tuple=True)
      (OutEnc, (ForwardState, BackwardState)) = tf.nn.bidirectional_dynamic_rnn(Forward, back, encoder_inputs, dtype=tf.float32, sequence_length=seq_len, swap_memory=True)
      OutEnc = tf.concat(axis=2, values=OutEnc) # concatenate the forwards and backwards states
    return OutEnc, ForwardState, BackwardState


  def _reduce_states(self, ForwardState, BackwardState):
    """Add to the graph a linear layer to reduce the encoder's final FW and BW state into a single initial state for the decoder. This is needed because the encoder is bidirectional but the decoder is not.

    Args:
      ForwardState: LSTMStateTuple with hidden_dim units.
      BackwardState: LSTMStateTuple with hidden_dim units.

    Returns:
      state: LSTMStateTuple with hidden_dim units.
    """
    hidden_dim = self._hps.hidden_dim
    with tf.variable_scope('reduce_final_st'):

      #Define weights and biases to reduce the cell and reduce the state
      
      RedCBias = tf.get_variable('bias_reduce_c', [hidden_dim], dtype=tf.float32, initializer=self.trunc_norm_init)
      RedHBias = tf.get_variable('bias_reduce_h', [hidden_dim], dtype=tf.float32, initializer=self.trunc_norm_init)
      WtHReduced = tf.get_variable('w_reduce_h', [hidden_dim * 2, hidden_dim], dtype=tf.float32, initializer=self.trunc_norm_init)
      WtCReduced = tf.get_variable('w_reduce_c', [hidden_dim * 2, hidden_dim], dtype=tf.float32, initializer=self.trunc_norm_init)

      # Apply linear layer
      h = tf.concat(axis=1, values=[ForwardState.h, BackwardState.h]) # Concatenation of fw and bw state
      c = tf.concat(axis=1, values=[ForwardState.c, BackwardState.c]) # Concatenation of fw and bw cell
      Hmodified = tf.nn.relu(tf.matmul(h, WtHReduced) + RedHBias) # Get new state from old state
      Cmodified= tf.nn.relu(tf.matmul(c, WtCReduced ) + RedCBias) # Get new cell from old cell
      
      return tf.contrib.rnn.LSTMStateTuple(Cmodified, Hmodified) # Return new cell and state


  def _add_decoder(self, inputs):
    """Add attention decoder to the graph. In train or eval mode, you call this once to get output on ALL steps. In decode (beam search) mode, you call this once for EACH decoder step.

    Args:
      inputs: inputs to the decoder (word embeddings). A list of tensors shape (batch_size, emb_dim)

    Returns:
      outputs: List of tensors; the outputs of the decoder
      out_state: The final state of the decoder
      attn_dists: A list of tensors; the attention distributions
      p_gens: A list of tensors shape (batch_size, 1); the generation probabilities
      coverage: A tensor, the current coverage vector
    """
    hps = self._hps
    cell = tf.contrib.rnn.LSTMCell(hps.hidden_dim, state_is_tuple=True, initializer=self.rand_unif_init)

    prev_coverage = self.prev_coverage if hps.mode=="decode" and hps.coverage else None # In decode mode, we run attention_decoder one step at a time and so need to pass in the previous step's coverage vector each time

    outputs, out_state, attn_dists, p_gens, coverage = attention_decoder(inputs, self._dec_in_state, self._enc_states, self._EncoderMask, cell, initial_state_attention=(hps.mode=="decode"), pointer_gen=hps.pointer_gen, use_coverage=hps.coverage, prev_coverage=prev_coverage)

    return outputs, out_state, attn_dists, p_gens, coverage

  def _calc_final_dist(self, vocab_dists, attn_dists):
    """Calculate the final distribution, for the pointer-generator model

    Args:
      vocab_dists: The vocabulary distributions. List length max_dec_steps of (batch_size, vsize) arrays. The words are in the order they appear in the vocabulary file.
      attn_dists: The attention distributions. List length max_dec_steps of (batch_size,  LengthOfAttention) arrays

    Returns:
      DistributionFinal: The final distributions. List length max_dec_steps of (batch_size, VSzEXt) arrays.
    """
    with tf.variable_scope('final_distribution'):
      # Multiply vocab dists by p_gen and attention dists by (1-p_gen)
      
      attn_dists = [(1-p_gen) * dist for (p_gen,dist) in zip(self.p_gens, attn_dists)]
      vocab_dists = [p_gen * dist for (p_gen,dist) in zip(self.p_gens, vocab_dists)]

      # Concatenate some zeros to each vocabulary dist, to hold the probabilities for in-article OOV words
      VSzEXt = self._vocab.size() + self._MaximumOOvs # the maximum (over the batch) size of the extended vocabulary
      ZeroExt = tf.zeros((self._hps.batch_size, self._MaximumOOvs))
      EXtendedVocabulary = [tf.concat(axis=1, values=[dist, ZeroExt]) for dist in vocab_dists] # list length max_dec_steps of shape (batch_size,VSzEXt)

      # Project the values in the attention distributions onto the appropriate entries in the final distributions
      # This means that if a_i = 0.1 and the ith encoder word is w, and w has index 500 in the vocabulary, then we add 0.1 onto the 500th entry of the final distribution
      # This is done for each decoder timestep.
      # This is fiddly; we use tf.scatter_nd to do the projection
      NumberOfBatches = tf.range(0, limit=self._hps.batch_size) # shape (batch_size)
      NumberOfBatches = tf.expand_dims(NumberOfBatches, 1) # shape (batch_size, 1)
      LengthOfAttention = tf.shape(self._BatchEncoder_extend_vocab)[1] # number of states we attend over
      NumberOfBatches = tf.tile(NumberOfBatches, [1, LengthOfAttention]) # shape (batch_size,  LengthOfAttention)
      ind = tf.stack( (NumberOfBatches, self._BatchEncoder_extend_vocab), axis=2) # shape (batch_size, enc_t, 2)
      shape = [self._hps.batch_size, VSzEXt]
      ProjectedAttention = [tf.scatter_nd(ind, copy_dist, shape) for copy_dist in attn_dists] # list length max_dec_steps (batch_size, VSzEXt)

      # Add the vocab distributions and the copy distributions together to get the final distributions
      # final_dists is a list length max_dec_steps; each entry is a tensor shape (batch_size, VSzEXt) giving the final distribution for that decoder timestep
      # Note that for decoder timesteps and examples corresponding to a [PAD] token, this is junk - ignore.
      DistributionFinal = [vocab_dist + copy_dist for (vocab_dist,copy_dist) in zip(EXtendedVocabulary, ProjectedAttention)]

      return DistributionFinal

  def _add_emb_vis(self, embedding_var):
 
    """Make the vocab metadata file, then make the projector config file pointing to it."""
    DirectoryTrain = os.path.join(FLAGS.log_root, "train")
    PathOfMetadata = os.path.join(DirectoryTrain, "vocab_metadata.tsv")
    self._vocab.write_metadata(PathOfMetadata) # write metadata file
    Summ = tf.summary.FileWriter(DirectoryTrain)
    Configuration = projector.ProjectorConfig()
    embedding = Configuration.embeddings.add()
    embedding.tensor_name = embedding_var.name
    embedding.metadata_path = PathOfMetadata
    projector.visualize_embeddings(Summ, Configuration)

  def _add_seq2seq(self):
    """Add the whole sequence-to-sequence model to the graph."""
   
    vsize = self._vocab.size() # size of the vocabulary
    hps = self._hps
    

    with tf.variable_scope('seq2seq'):
      # Some initializers
      
      self.trunc_norm_init = tf.truncated_normal_initializer(stddev=hps.trunc_norm_init_std)
      self.rand_unif_init = tf.random_uniform_initializer(-hps.rand_unif_init_mag, hps.rand_unif_init_mag, seed=123)
      

      # Add embedding matrix (shared by the encoder and decoder inputs)
      with tf.variable_scope('embedding'):
        embedding = tf.get_variable('embedding', [vsize, hps.emb_dim], dtype=tf.float32, initializer=self.trunc_norm_init)
        if hps.mode=="train": self._add_emb_vis(embedding) # add to tensorboard
        EncEmbeddedInputs = tf.nn.embedding_lookup(embedding, self._BatchEncoder) # tensor with shape (batch_size, max_enc_steps, emb_size)
        DEcEmbeddedInputs = [tf.nn.embedding_lookup(embedding, x) for x in tf.unstack(self._BDecoder, axis=1)] # list length max_dec_steps containing shape (batch_size, emb_size)

      # Add the encoder.
      enc_outputs, ForwardState, BackwardState = self._add_encoder(EncEmbeddedInputs, self._LensEncoder)
      self._enc_states = enc_outputs

      # Our encoder is bidirectional and our decoder is unidirectional so we need to reduce the final encoder hidden state to the right size to be the initial decoder hidden state
      self._dec_in_state = self._reduce_states(ForwardState, BackwardState)

      # Add the decoder.
      with tf.variable_scope('decoder'):
        decoder_outputs, self._dec_out_state, self.attn_dists, self.p_gens, self.coverage = self._add_decoder(DEcEmbeddedInputs)

      # Add the output projection to obtain the vocabulary distribution
      with tf.variable_scope('output_projection'):
        w = tf.get_variable('w', [hps.hidden_dim, vsize], dtype=tf.float32, initializer=self.trunc_norm_init)
        w_t = tf.transpose(w)
        v = tf.get_variable('v', [vsize], dtype=tf.float32, initializer=self.trunc_norm_init)
        Scores = [] # Scores is the vocabulary distribution before applying softmax. Each entry on the list corresponds to one decoder step
        for hh,output in enumerate(decoder_outputs):
          if hh > 0:
            tf.get_variable_scope().reuse_variables()
          Scores.append(tf.nn.xw_plus_b(output, w, v)) # apply the linear layer

        vocab_dists = [tf.nn.softmax(s) for sss in Scores] # The vocabulary distributions. List length max_dec_steps of (batch_size, vsize) arrays. The words are in the order they appear in the vocabulary file.


      # For pointer-generator model, calc final distribution from copy distribution and vocabulary distribution
      if FLAGS.pointer_gen:
        DistributionFinal = self._calc_final_dist(vocab_dists, self.attn_dists)
      else: # final distribution is just vocabulary distribution
        DistributionFinal = vocab_dists



      if hps.mode in ['train', 'eval']:
        # Calculate the loss
        with tf.variable_scope('loss'):
          if FLAGS.pointer_gen:
            # Calculate the loss per step
            # This is fiddly; we use tf.gather_nd to pick out the probabilities of the gold target words
            EachStepLoss = [] # will be list length max_dec_steps containing shape (batch_size)
            NumberOfBatches = tf.range(0, limit=hps.batch_size) # shape (batch_size)
            for dec_step, dist in enumerate(DistributionFinal):
              targets = self._BatchTar[:,dec_step] # The indices of the target words. shape (batch_size)
              ind = tf.stack( (NumberOfBatches, targets), axis=1) # shape (batch_size, 2)
              gold = tf.gather_nd(dist, ind) # shape (batch_size). prob of correct words on this step
              l = -tf.log(gold)
              EachStepLoss.append(l)

            # Apply dec_padding_mask and get loss
            self._loss = _mask_and_avg(EachStepLoss, self._MaskDecoderPad)

          else: # baseline model
            self._loss = tf.contrib.seq2seq.sequence_loss(tf.stack(Scores, axis=1), self._BatchTar, self._MaskDecoderPad) # this applies softmax internally

          tf.summary.scalar('loss', self._loss)

          # Calculate coverage loss from the attention distributions
          if hps.coverage:
            with tf.variable_scope('coverage_loss'):
              self._coverage_loss = _coverage_loss(self.attn_dists, self._MaskDecoderPad)
              tf.summary.scalar('coverage_loss', self._coverage_loss)
            self._total_loss = self._loss + hps.cov_loss_wt * self._coverage_loss
            tf.summary.scalar('total_loss', self._total_loss)

    if hps.mode == "decode":
      # We run decode beam search mode one decoder step at a time
      assert len(DistributionFinal)==1 # DistributionFinal is a singleton list containing shape (batch_size, VSzEXt)
      DistributionFinal = DistributionFinal[0]
      topk_probs, self._topk_ids = tf.nn.top_k(DistributionFinal, hps.batch_size*2) # take the k largest probs. note batch_size=beam_size in decode mode
      self._topk_log_probs = tf.log(topk_probs)


  def _add_train_op(self):
    """Sets self._train_op, the op to run for training."""
    # Take gradients of the trainable variables w.r.t. the loss function to minimize
    MinimizingLoss = self._total_loss if self._hps.coverage else self._loss
    ttV = tf.trainable_variables()
    gradients = tf.gradients(MinimizingLoss , ttV, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)

    # Clip the gradients
    with tf.device("/gpu:0"):
      grads, global_norm = tf.clip_by_global_norm(gradients, self._hps.max_grad_norm)

    # Add a summary
    tf.summary.scalar('global_norm', global_norm)

    # Apply adagrad optimizer
    optimizer = tf.train.AdagradOptimizer(self._hps.lr, initial_accumulator_value=self._hps.adagrad_init_acc)
    with tf.device("/gpu:0"):
      self._train_op = optimizer.apply_gradients(zip(grads, ttV), global_step=self.global_step, name='train_step')


  def build_graph(self):
    """Add the placeholders, model, global step, train_op and summaries to the graph"""
    tf.logging.info('Building graph...')
    t0 = time.time()
    self._add_placeholders()
    with tf.device("/gpu:0"):
      self._add_seq2seq()
    self.global_step = tf.Variable(0, name='global_step', trainable=False)
    if self._hps.mode == 'train':
      self._add_train_op()
    self._summaries = tf.summary.merge_all()
    t1 = time.time()
    tf.logging.info('Time to build graph: %i seconds', t1 - t0)

  def run_train_step(self, sess, batch):
    """Runs one training iteration. Returns a dictionary containing train op, summaries, loss, global_step and (optionally) coverage loss."""
    FD = self._make_feed_dict(batch)
    to_return = {
        'train_op': self._train_op,
        'summaries': self._summaries,
        'loss': self._loss,
        'global_step': self.global_step,
    }
    if self._hps.coverage:
      to_return['coverage_loss'] = self._coverage_loss
    return sess.run(to_return, FD)

  def run_eval_step(self, sess, batch):
    """Runs one evaluation iteration. Returns a dictionary containing summaries, loss, global_step and (optionally) coverage loss."""
    FD = self._make_feed_dict(batch)
    to_return = {
        'summaries': self._summaries,
        'loss': self._loss,
        'global_step': self.global_step,
    }
    if self._hps.coverage:
      to_return['coverage_loss'] = self._coverage_loss
    return sess.run(to_return, FD)

  def run_encoder(self, sess, batch):
    """For beam search decoding. Run the encoder on the batch and return the encoder states and decoder initial state.

    Args:
      sess: Tensorflow session.
      batch: Batch object that is the same example repeated across the batch (for beam search)

    Returns:
      enc_states: The encoder states. A tensor of shape [batch_size, <=max_enc_steps, 2*hidden_dim].
      dec_in_state: A LSTMStateTuple of shape ([1,hidden_dim],[1,hidden_dim])
    """
    FD = self._make_feed_dict(batch, just_enc=True) # feed the batch into the placeholders
    (enc_states, dec_in_state, global_step) = sess.run([self._enc_states, self._dec_in_state, self.global_step], FD) # run the encoder

    # dec_in_state is LSTMStateTuple shape ([batch_size,hidden_dim],[batch_size,hidden_dim])
    # Given that the batch is a single example repeated, dec_in_state is identical across the batch so we just take the top row.
    dec_in_state = tf.contrib.rnn.LSTMStateTuple(dec_in_state.c[0], dec_in_state.h[0])
    return enc_states, dec_in_state


  def decode_onestep(self, sess, batch, latest_tokens, enc_states, dec_init_states, prev_coverage):
    """For beam search decoding. Run the decoder for one step.

    Args:
      sess: Tensorflow session.
      batch: Batch object containing single example repeated across the batch
      latest_tokens: Tokens to be fed as input into the decoder for this timestep
      enc_states: The encoder states.
      dec_init_states: List of beam_size LSTMStateTuples; the decoder states from the previous timestep
      prev_coverage: List of np arrays. The coverage vectors from the previous timestep. List of None if not using coverage.

    Returns:
      ids: top 2k ids. shape [beam_size, 2*beam_size]
      probs: top 2k log probabilities. shape [beam_size, 2*beam_size]
      StateNewNEw: new states of the decoder. a list length beam_size containing
        LSTMStateTuples each of shape ([hidden_dim,],[hidden_dim,])
      attn_dists: List length beam_size containing lists length attn_length.
      p_gens: Generation probabilities for this step. A list length beam_size. List of None if in baseline mode.
      new_coverage: Coverage vectors for this step. A list of arrays. List of None if coverage is not turned on.
    """

    BSize = len(dec_init_states)

    #Turn dec_init_states (a list of LSTMStateTuples) into a single LSTMStateTuple for the batch
    hiddens = [np.expand_dims(state.h, axis=0) for state in dec_init_states]
    c = [np.expand_dims(state.c, axis=0) for state in dec_init_states]
    Hmodified = np.concatenate(hiddens, axis=0)  # shape [batch_size,hidden_dim]
    Cmodified = np.concatenate(c, axis=0)  # shape [batch_size,hidden_dim]
    new_dec_in_state = tf.contrib.rnn.LSTMStateTuple(Cmodified, Hmodified)

    to_return = {
      "ids": self._topk_ids,
      "probs": self._topk_log_probs,
      "states": self._dec_out_state,
      "attn_dists": self.attn_dists
    }
    feed = {
        self._enc_states: enc_states,
        self._EncoderMask: batch.enc_padding_mask,
        self._dec_in_state: new_dec_in_state,
        self._BDecoder: np.transpose(np.array([latest_tokens])),
    }

    

    if FLAGS.pointer_gen:
      feed[self._BatchEncoder_extend_vocab] = batch.enc_batch_extend_vocab
      feed[self._MaximumOOvs] = batch.max_art_oovs
      to_return['p_gens'] = self.p_gens

    if self._hps.coverage:
      feed[self.prev_coverage] = np.stack(prev_coverage, axis=0)
      to_return['coverage'] = self.coverage

    final = sess.run(to_return, FD=feed) # run the decoder step

    # Convert final['states'] (a single LSTMStateTuple) into a list of LSTMStateTuple -- one for each hypothesis
    StateNewNEw = [tf.contrib.rnn.LSTMStateTuple(final['states'].c[i, :], final['states'].h[i, :]) for i in xrange(BSize)]

    # Convert singleton list containing a tensor to a list of k arrays
    assert len(final['attn_dists'])==1
    attn_dists = final['attn_dists'][0].tolist()

    if FLAGS.pointer_gen:
      # Convert singleton list containing a tensor to a list of k arrays
      assert len(final['p_gens'])==1
      p_gens = final['p_gens'][0].tolist()
    else:
      p_gens = [None for _ in xrange(BSize)]

    # Convert the coverage tensor to a list length k containing the coverage vector for each hypothesis
    if FLAGS.coverage:
      new_coverage = final['coverage'].tolist()
      assert len(new_coverage) == BSize
    else:
      new_coverage = [None for _ in xrange(BSize)]

    return final['ids'], final['probs'], StateNewNEw, attn_dists, p_gens, new_coverage


def _mask_and_avg(values, padding_mask):
  """Applies mask to values then returns overall average (a scalar)

  Args:
    values: a list length max_dec_steps containing arrays shape (batch_size).
    padding_mask: tensor shape (batch_size, max_dec_steps) containing 1s and 0s.

  Returns:
    a scalar
  """

  dl = tf.reduce_sum(padding_mask, axis=1) # shape batch_size. float32
  Each_Step_Val = [v * padding_mask[:,dec_step] for dec_step,v in enumerate(values)]
  ForEveryExVal = sum(Each_Step_Val)/dl # shape (batch_size); normalized value for each batch member
  return tf.reduce_mean(ForEveryExVal) # overall average


def _coverage_loss(attn_dists, padding_mask):
  """Calculates the coverage loss from the attention distributions.

  Args:
    attn_dists: The attention distributions for each decoder timestep. A list length max_dec_steps containing shape (batch_size, attn_length)
    padding_mask: shape (batch_size, max_dec_steps).

  Returns:
    coverage_loss: scalar
  """
  LossesInCoverage = [] # Coverage loss per decoder timestep. Will be list length max_dec_steps containing shape (batch_size).
  coverage = tf.zeros_like(attn_dists[0]) # shape (batch_size, attn_length). Initial coverage is zero.
  for b in attn_dists:
    c_ls = tf.reduce_sum(tf.minimum(b, coverage), [1]) # calculate the coverage loss for this step
    LossesInCoverage.append(c_ls)
    coverage += b # update the coverage vector
  coverage_loss = _mask_and_avg(LossesInCoverage , padding_mask)
  return coverage_loss
