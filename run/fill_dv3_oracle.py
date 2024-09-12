import warnings
from functools import partial as bind
import sys
sys.path.append("./ext/dreamerv3")
import dreamerv3
import embodied

warnings.filterwarnings('ignore', '.*truncated to dtype int32.*')

import re
from collections import defaultdict
from functools import partial as bind

import embodied
import numpy as np

def main():
  prefill_oracle = 80
  config = embodied.Config(dreamerv3.Agent.configs['defaults'])
  config = config.update({
      **dreamerv3.Agent.configs['size100m'],
      'logdir': f'./logdir/{embodied.timestamp()}-example',
      'run.train_ratio': 32,
      'enc.spaces': 'image',
      'dec.spaces': 'image',
      'replay.size': 6e5,
      # 'run.num_envs': 1,
      # 'run.num_envs_eval': 1,
      'run.driver_parallel': True,
  })
  if prefill_oracle > 0:
    config = config.update({
        'run.num_envs': 1,
        'run.num_envs_eval': 1,
        'run.driver_parallel': False,
        'jax.platform': 'cpu',
    })

  config = embodied.Flags(config).parse()

  print('Logdir:', config.logdir)
  logdir = embodied.Path(config.logdir)
  logdir.mkdir()
  config.save(logdir / 'config.yaml')

  def make_agent(config):
    env = make_env(config)
    dv3_agnt = dreamerv3.Agent(env.obs_space, env.act_space, config)
    agent = RandomAgent(env.obs_space, env.act_space)
    env.close()
    return agent

  def make_logger(config):
    logdir = embodied.Path(config.logdir)
    return embodied.Logger(embodied.Counter(), [
        embodied.logger.TerminalOutput(config.filter),
        embodied.logger.JSONLOutput(logdir, 'metrics.jsonl'),
        embodied.logger.TensorBoardOutput(logdir),
        # embodied.logger.WandbOutput(logdir.name, config=config),
    ])

  def make_replay(config):
    return embodied.replay.Replay(
        length=config.batch_length,
        capacity=config.replay.size,
        directory=embodied.Path(config.logdir) / 'replay',
        online=config.replay.online)

  def make_env(config, env_id=0):
    from rgbd_sym.api import make_env 
    from embodied.envs import from_gym
    from embodied.core.wrappers import ResizeImage 
    original_env, env_config = make_env(tags=[], seed=0)
    env = from_gym.FromGym(original_env)
    env = ResizeImage(env)
    env = dreamerv3.wrap_env(env, config)
    return env

  args = embodied.Config(
      **config.run,
      logdir=config.logdir,
      batch_size=config.batch_size,
      batch_length=config.batch_length,
      batch_length_eval=config.batch_length_eval,
      replay_context=config.replay_context,
  )

  train(
      bind(make_agent, config),
      bind(make_replay, config),
      bind(make_env, config),
      bind(make_logger, config), args, prefill_oracle, config)





def train(make_agent, make_replay, make_env, make_logger, args, prefill_oracle, config):

  # agent = make_agent()
  replay = make_replay()
  logger = make_logger()

  logdir = embodied.Path(args.logdir)
  logdir.mkdir()
  print('Logdir', logdir)
  step = logger.step
  usage = embodied.Usage(**args.usage)
  agg = embodied.Agg()
  epstats = embodied.Agg()
  episodes = defaultdict(embodied.Agg)
  policy_fps = embodied.FPS()
  train_fps = embodied.FPS()

  batch_steps = args.batch_size * (args.batch_length - args.replay_context)
  should_expl = embodied.when.Until(args.expl_until)
  should_train = embodied.when.Ratio(args.train_ratio / batch_steps)
  should_log = embodied.when.Clock(args.log_every)
  should_eval = embodied.when.Clock(args.eval_every)
  should_save = embodied.when.Clock(args.save_every)

  @embodied.timer.section('log_step')
  def log_step(tran, worker):

    episode = episodes[worker]
    episode.add('score', tran['reward'], agg='sum')
    episode.add('length', 1, agg='sum')
    episode.add('rewards', tran['reward'], agg='stack')

    if tran['is_first']:
      episode.reset()

    if worker < args.log_video_streams:
      for key in args.log_keys_video:
        if key in tran:
          episode.add(f'policy_{key}', tran[key], agg='stack')
    for key, value in tran.items():
      if re.match(args.log_keys_sum, key):
        episode.add(key, value, agg='sum')
      if re.match(args.log_keys_avg, key):
        episode.add(key, value, agg='avg')
      if re.match(args.log_keys_max, key):
        episode.add(key, value, agg='max')

    if tran['is_last']:
      result = episode.result()
      logger.add({
          'score': result.pop('score'),
          'length': result.pop('length'),
      }, prefix='episode')
      rew = result.pop('rewards')
      if len(rew) > 1:
        result['reward_rate'] = (np.abs(rew[1:] - rew[:-1]) >= 0.01).mean()
      epstats.add(result)

  fns = [bind(make_env, i) for i in range(args.num_envs)]
  driver = embodied.Driver(fns, args.driver_parallel)
  driver.on_step(lambda tran, _: step.increment())
  driver.on_step(lambda tran, _: policy_fps.step())
  driver.on_step(replay.add)
  driver.on_step(log_step)


  if prefill_oracle > 0:
    _env= driver.envs[0]
    dv3_agnt = dreamerv3.Agent(_env.obs_space, _env.act_space, config)
    oracle_agent = RandomAgent(driver.envs[0].obs_space, driver.envs[0].act_space, dv3_agnt) 
    driver.reset(oracle_agent.init_policy)
    for i in range(prefill_oracle):
      driver(oracle_agent.policy, episodes=1)
      print("eps / total: ", i+1, "/", prefill_oracle)

  checkpoint = embodied.Checkpoint(logdir / 'checkpoint.ckpt')
  checkpoint.step = step
  checkpoint.agent = dv3_agnt
  checkpoint.replay = replay
  checkpoint.save()

  # print('Start training loop')
  # policy = lambda *args: agent.policy(
  #     *args, mode='explore' if should_expl(step) else 'train')
  # driver.reset(agent.init_policy)
  # while step < args.steps:

  #   driver(policy, steps=10)

  #   if should_eval(step) and len(replay):
  #     mets, _ = agent.report(next(dataset_report), carry_report)
  #     logger.add(mets, prefix='report')

  #   if should_log(step):
  #     logger.add(agg.result())
  #     logger.add(epstats.result(), prefix='epstats')
  #     logger.add(embodied.timer.stats(), prefix='timer')
  #     logger.add(replay.stats(), prefix='replay')
  #     logger.add(usage.stats(), prefix='usage')
  #     logger.add({'fps/policy': policy_fps.result()})
  #     logger.add({'fps/train': train_fps.result()})
  #     logger.write()

  #   if should_save(step):
  #     checkpoint.save()

  logger.close()

class RandomAgent:

  def __init__(self, obs_space, act_space, dv3_agnt):
    self.obs_space = obs_space
    self.act_space = act_space
    self._dv3_agnt = dv3_agnt

  def init_policy(self, batch_size):
    return self._dv3_agnt.init_policy(batch_size)

  def init_train(self, batch_size):
    return ()

  def init_report(self, batch_size):
    return ()

  def policy(self, obs, carry=(), mode='train'):
    batch_size = len(obs['is_first'])
    act = {
        k: np.stack([v.sample() for _ in range(batch_size)])
        for k, v in self.act_space.items() if k != 'reset'}
    
    outs = {}
    outs['deter'] = carry[0]['deter']
    outs['stoch'] = carry[0]['stoch']
    return act, outs, carry

  def train(self, data, carry=()):
    outs = {}
    metrics = {}
    return outs, carry, metrics

  def report(self, data, carry=()):
    report = {}
    return report, carry

  def dataset(self, generator):
    return generator()

  def save(self):
    return None

  def load(self, data=None):
    pass

class OracleAgent:

  def __init__(self, obs_space, act_space, original_env):
    self.obs_space = obs_space
    self.act_space = act_space
    self._original_env = original_env

  def init_policy(self, batch_size):
    return ()

  def init_train(self, batch_size):
    return ()

  def init_report(self, batch_size):
    return ()


  def policy(self, obs, carry=(), mode='train'):
    batch_size = len(obs['is_first'])
    assert batch_size == 1

    act = {
        k: np.stack([self._original_env[i].get_oracle_action().astype(np.float32) for i in range(batch_size)])
        for k, v in self.act_space.items() if k != 'reset'}
        
    outs = {}

    return act, outs, carry

  def train(self, data, carry=()):
    outs = {}
    metrics = {}
    return outs, carry, metrics

  def report(self, data, carry=()):
    report = {}
    return report, carry

  def dataset(self, generator):
    return generator()

  def save(self):
    return None

  def load(self, data=None):
    pass


if __name__ == '__main__':
  main()
