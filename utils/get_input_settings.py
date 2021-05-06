
def get_input_settings(globs):
  """
  globs: focus.globals
  return the settings from the 
  input file as a dict.

  To use:
    import focus.globals as globs
    d = get_input_settings(globs)
  """
  d = {}
  d['nseg']           = globs.nseg
  d['nteta']          = globs.nteta       
  d['nzeta']          = globs.nzeta       
  d['nfcoil']         = globs.nfcoil      
  d['ncoils']         = globs.ncoils      
  d['curv_alpha']     = globs.curv_alpha  
  d['weight_bnorm']   = globs.weight_bnorm
  d['weight_bharm']   = globs.weight_bharm
  d['weight_tflux']   = globs.weight_tflux
  d['target_tflux']   = globs.target_tflux
  d['weight_ttlen']   = globs.weight_ttlen
  d['target_length']  = globs.target_length
  d['case_length']    = globs.case_length 
  d['case_curv']      = globs.case_curv   
  d['case_bnormal']   = globs.case_bnormal
  d['isnormweight']   = globs.isnormweight
  d['isnormalize']    = globs.isnormalize 
  d['isvarycurrent']  = globs.isvarycurrent
  d['isvarygeometry'] = globs.isvarygeometry
  d['case_init']      = globs.case_init
  d['case_coils']     = globs.case_coils
  d['case_surface']   = globs.case_surface
  d['init_current']   = globs.init_current
  d['init_radius']    = globs.init_radius
  d['issymmetric']    = globs.issymmetric
  return d
