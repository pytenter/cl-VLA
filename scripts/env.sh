#!/usr/bin/env bash
# Source one profile in a fresh, already activated model environment.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  printf '%s\n' 'Use: source scripts/env.sh <profile>' >&2
  exit 2
fi
_cl_vla_requested="${1:-libero-fourway}"
case "$_cl_vla_requested" in
  libero-fourway|pi05-libero-fourway) _cl_vla_profile=libero-fourway; _cl_vla_family=pi05; _cl_vla_arg=libero-fourway;;
  libero-plain-prototype|pi05-libero-plain-prototype) _cl_vla_profile=libero-plain-prototype; _cl_vla_family=pi05; _cl_vla_arg=libero-plain-prototype;;
  kinova|pi05-kinova) _cl_vla_profile=kinova; _cl_vla_family=pi05; _cl_vla_arg=kinova;;
  openvla-libero|libero) _cl_vla_profile=openvla-libero; _cl_vla_family=openvla; _cl_vla_arg=libero;;
  openvla-robotwin|robotwin) _cl_vla_profile=openvla-robotwin; _cl_vla_family=openvla; _cl_vla_arg=robotwin;;
  *)
    printf '%s\n' 'Profiles: libero-fourway | libero-plain-prototype | kinova | openvla-libero | openvla-robotwin' >&2
    unset _cl_vla_requested
    return 2;;
esac
if [[ -n "${CL_VLA_ACTIVE_PROFILE:-}" && "$CL_VLA_ACTIVE_PROFILE" != "$_cl_vla_profile" ]]; then
  printf 'Current profile is %s. Open a fresh terminal and activate the environment for %s.\n' "$CL_VLA_ACTIVE_PROFILE" "$_cl_vla_profile" >&2
  unset _cl_vla_requested _cl_vla_profile _cl_vla_family _cl_vla_arg
  return 2
fi
_cl_vla_scripts="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ "$_cl_vla_family" == pi05 ]]; then
  source "$_cl_vla_scripts/env_pi05.sh" "$_cl_vla_arg"
else
  source "$_cl_vla_scripts/env_openvla.sh" "$_cl_vla_arg"
fi
_cl_vla_status=$?
if [[ "$_cl_vla_status" -ne 0 ]]; then
  unset _cl_vla_requested _cl_vla_profile _cl_vla_family _cl_vla_arg _cl_vla_scripts
  return "$_cl_vla_status"
fi
export CL_VLA_ACTIVE_PROFILE="$_cl_vla_profile" CL_VLA_ACTIVE_FAMILY="$_cl_vla_family"
unset _cl_vla_requested _cl_vla_profile _cl_vla_family _cl_vla_arg _cl_vla_scripts _cl_vla_status
