import { Component } from 'react'

export default class Boundary extends Component {
  state = { error: null }

  static getDerivedStateFromError(error) {
    return { error }
  }

  render() {
    if (!this.state.error) return this.props.children
    return (
      <div className="lab">
        <main className="lab__main">
          <section className="controls">
            <h3>Something went wrong in the Arena</h3>
            <p className="err">{String(this.state.error.message || this.state.error)}</p>
            <div className="actions">
              <button className="go" onClick={() => window.location.reload()}>
                Reload the Arena
              </button>
              <a className="mini" href="#/">
                Back to site
              </a>
            </div>
          </section>
        </main>
      </div>
    )
  }
}
