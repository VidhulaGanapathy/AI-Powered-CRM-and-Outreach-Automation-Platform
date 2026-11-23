// @vitest-environment jsdom
import React from 'react'
import { describe, it, expect, afterEach } from 'vitest'
import { render, screen, within, cleanup } from '@testing-library/react'
import History from '../History'

afterEach(() => {
  cleanup()
})

describe('History', () => {
  it('renders the page header and audit table columns', () => {
    render(<History />)

    expect(screen.getByText('Enrichment History')).toBeTruthy()
    expect(
      screen.getByText('Audit history of previous enrichment lookups.')
    ).toBeTruthy()
    expect(screen.getByText('Requester')).toBeTruthy()
    expect(screen.getByText('Timestamp')).toBeTruthy()
    expect(screen.getByText('Provider')).toBeTruthy()
    expect(screen.getByText('Outcome')).toBeTruthy()
    expect(screen.getByText('Credits')).toBeTruthy()
    expect(screen.getByText('Email Provenance')).toBeTruthy()
  })

  it('renders the requester and formatted timestamp in the data row', () => {
    render(<History />)

    const [, ...dataRows] = screen.getAllByRole('row')
    const expectedTimestamp = new Date('2026-10-30T10:30:00').toLocaleString()

    expect(dataRows).toHaveLength(1)
    expect(within(dataRows[0]).getByText('requester 1')).toBeTruthy()
    expect(within(dataRows[0]).getByText(expectedTimestamp)).toBeTruthy()
  })

  it('renders exactly one data row with Apollo, Success, and LinkedIn', () => {
    render(<History />)

    const [, ...dataRows] = screen.getAllByRole('row')
    expect(dataRows).toHaveLength(1)

    const row = within(dataRows[0])
    expect(row.getByText('Apollo')).toBeTruthy()
    expect(row.getByText('Success')).toBeTruthy()
    expect(row.getByText('LinkedIn')).toBeTruthy()
  })

  it('renders the credits value inside the enrichment row', () => {
    render(<History />)

    const [, ...dataRows] = screen.getAllByRole('row')
    expect(dataRows).toHaveLength(1)

    const cells = within(dataRows[0]).getAllByRole('cell')
    expect(cells[4].textContent).toBe('2')
    expect(within(dataRows[0]).getByText('2')).toBeTruthy()
  })

  it('shows single-page pagination summary with navigation buttons disabled', () => {
    render(<History />)

    expect(screen.queryByText('No enrichment history found')).toBeNull()
    expect(screen.getByText('Showing 1 record(s)')).toBeTruthy()
    expect(screen.getByText('Page 1 of 1')).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Previous' }).disabled).toBe(true)
    expect(screen.getByRole('button', { name: 'Next' }).disabled).toBe(true)
  })
})
