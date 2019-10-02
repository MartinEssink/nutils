import sys, os, tempfile, io
from nutils import cli, log, testing

def main(
  iarg: 'integer' = 1,
  farg: 'float' = 1.,
  sarg: 'string' = 'foo'):
  assert isinstance(iarg, int), 'n should be int, got {}'.format(type(iarg))
  assert isinstance(farg, float), 'f should be float, got {}'.format(type(farg))
  assert isinstance(sarg, str), 'f should be str, got {}'.format(type(sarg))
  print('all OK')

@testing.parametrize
class run(testing.TestCase):

  scriptname = 'test.py'

  def setUpContext(self, stack):
    super().setUpContext(stack)
    self.outrootdir = stack.enter_context(tempfile.TemporaryDirectory())

  def _cli(self, *args, funcname='main'):
    _savestreams = sys.stdout, sys.stderr
    _saveargs = tuple(sys.argv)
    try:
      sys.stdout = sys.stderr = stringio = io.StringIO()
      sys.argv[:] = self.scriptname,
      if self.method == 'choose' and funcname:
        sys.argv.append(funcname)
      sys.argv.extend(args)
      getattr(cli, self.method)(main, loaduserconfig=False)
    except SystemExit as e:
      status = e
    else:
      status = None
    finally:
      sys.stdout, sys.stderr = _savestreams
      sys.argv[:] = _saveargs
    return status, stringio.getvalue()

  def test_good(self):
    args = ['--outrootdir='+self.outrootdir, '--nopdb', '--iarg=1', '--farg=1', '--sarg=1']
    status, output = self._cli(*args)
    with self.subTest('outdir'):
      self.assertTrue(os.path.isdir(os.path.join(self.outrootdir,self.scriptname)), 'output directory not found')
    with self.subTest('argparse'):
      log.info(output)
      self.assertIn('all OK', output)
    with self.subTest('exitstatus'):
      self.assertIsNotNone(status)
      self.assertEqual(status.code, 0)

  def test_badarg(self):
    status, output = self._cli('--bla') if self.method == 'run' else self._cli(funcname='bla')
    with self.subTest('argparse'):
      log.info(output)
      self.assertNotIn('all OK', output)
    with self.subTest('exitstatus'):
      self.assertIsNotNone(status)
      self.assertEqual(status.code, 2)

  def test_badvalue(self):
    status, output = self._cli('--outrootdir='+self.outrootdir, '--nopdb', '--iarg=1', '--farg=x', '--sarg=1')
    with self.subTest('outdir'):
      self.assertFalse(os.path.isdir(os.path.join(self.outrootdir,self.scriptname)), 'outdir directory found')
    with self.subTest('argparse'):
      log.info(output)
      self.assertNotIn('all OK', output)
    with self.subTest('exitstatus'):
      self.assertIsNotNone(status)
      self.assertEqual(status.code, 2)

  def test_help(self):
    for arg in '-h', '--help':
      for funcname in ('main', False):
        status, output = self._cli(arg, funcname=funcname)
        with self.subTest(arg=arg, funcname=funcname, test='argparse'):
          self.assertEqual(output[:6], 'USAGE:')
        with self.subTest(arg=arg, funcname=funcname, test='exitstatus'):
          self.assertIsNotNone(status)
          self.assertEqual(status.code, 1)

run(method='run')
run(method='choose')
