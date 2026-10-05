async function call(path, options) {
  let res
  try {
    res = await fetch(path, options)
  } catch {
    throw new Error('Cannot reach the VelocityLLM engine. Check the Wi-Fi and the address.')
  }
  let body = null
  try {
    body = await res.json()
  } catch {
    body = null
  }
  if (!res.ok) {
    const detail = body && body.detail
    throw new Error(typeof detail === 'string' ? detail : detail ? JSON.stringify(detail) : `Engine returned HTTP ${res.status}`)
  }
  return body
}

const post = (path, data) => call(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(data || {}) })

export const liveState = (deviceId) => call(`/api/live/state${deviceId ? `?device_id=${encodeURIComponent(deviceId)}` : ''}`)
export const joinSession = (name) => post('/api/live/join', { name })
export const sendPrompts = (device_id, spec) => post('/api/live/send', { device_id, spec })
export const saveConfig = (config) => post('/api/live/config', config)
export const startNow = () => post('/api/live/start')
export const clearSpecs = () => post('/api/live/reset')
export const cancelRun = (id) => post(`/api/runs/${id}/cancel`)

export const store = {
  get(key) {
    try {
      return window.localStorage.getItem(key)
    } catch {
      return null
    }
  },
  set(key, value) {
    try {
      if (value === null) window.localStorage.removeItem(key)
      else window.localStorage.setItem(key, value)
    } catch {
      return
    }
  },
}
