import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { compactParams } from '../features/program/ProgramHero';
import { byPolicy, isResults, type Results } from '../shared/lib/results';
import { App } from './App';
import fixture from './fixture.json';

const results = fixture as unknown as Results;

function mockFetch(impl: () => Promise<Response>) {
  vi.stubGlobal('fetch', vi.fn(impl));
}

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe('App', () => {
  it('shows a loading state, then every section', async () => {
    mockFetch(async () => new Response(JSON.stringify(results)));
    render(<App />);
    expect(screen.getByRole('status', { name: 'Loading results' })).toBeTruthy();
    for (const name of ['Programs compiled per workload', 'Are the guards sound?', 'Thunder on the same calls', 'Release gate']) {
      expect(await screen.findByRole('heading', { level: 2, name })).toBeTruthy();
    }
  });

  it('shows an error with a working retry', async () => {
    let calls = 0;
    mockFetch(async () => {
      calls += 1;
      return calls === 1 ? new Response('no', { status: 500 }) : new Response(JSON.stringify(results));
    });
    render(<App />);
    fireEvent.click(await screen.findByRole('button', { name: 'Try again' }));
    expect(await screen.findByRole('heading', { level: 2, name: 'Release gate' })).toBeTruthy();
  });

  it('shows an empty state when there are no results', async () => {
    mockFetch(async () => new Response('', { status: 404 }));
    render(<App />);
    expect(await screen.findByText('No results yet')).toBeTruthy();
  });

  it('switching the workload changes the program shown', async () => {
    mockFetch(async () => new Response(JSON.stringify(results)));
    render(<App />);
    const before = (await screen.findByLabelText('Generated computation')).textContent;
    fireEvent.click(screen.getByRole('button', { name: 'Padded encoder batches' }));
    expect(screen.getByLabelText('Generated computation').textContent).not.toBe(before);
  });

  it('expands the listing', async () => {
    mockFetch(async () => new Response(JSON.stringify(results)));
    render(<App />);
    const more = await screen.findByRole('button', { name: /Show all \d+ lines/ });
    fireEvent.click(more);
    expect(more.getAttribute('aria-expanded')).toBe('true');
  });
});

describe('helpers', () => {
  it('reads the fixture', () => {
    expect(isResults(results)).toBe(true);
    expect(isResults({})).toBe(false);
    expect(byPolicy(results.rows ?? [], 'greedy_decode').symbolic?.compiles).toBe(1);
  });

  it('compacts the parameter list', () => {
    expect(compactParams('def computation(p0, p1, p2, p3, a0, s0):')).toBe('def computation(p0, …, p3, a0, s0):');
  });
});
