# AI Rules for VoiceRAG Application

## Tech Stack
- React 18 with TypeScript
- React Router for client-side routing (routes kept in src/App.tsx)
- Tailwind CSS for styling and layout
- shadcn/ui component library for pre-built UI components
- Lucide React for icons
- Radix UI primitives (accessed via shadcn/ui)
- Vite as the build tool
- ESLint and Prettier for code formatting

## Library Usage Guidelines
- **UI Components**: Use shadcn/ui components as the primary source for UI elements (buttons, forms, cards, etc.)
- **Icons**: Use Lucide React icons exclusively; import individual icons as needed
- **Styling**: Use Tailwind CSS utility classes for all styling; avoid custom CSS unless absolutely necessary
- **Routing**: Use React Router v6; keep all route definitions in src/App.tsx
- **State Management**: Use React's built-in state hooks (useState, useEffect, useContext) for local state; avoid external state libraries unless complexity demands it
- **API Calls**: Use fetch or axios for HTTP requests; consider React Query for complex data fetching scenarios
- **Forms**: Use React Hook Form for form handling and validation when using shadcn/ui form components
- **Utilities**: Create reusable utility functions in src/utils/; avoid duplicating logic

## Project Structure
- **Pages**: Place page components in src/pages/ (each page corresponds to a route)
- **Components**: Place reusable components in src/components/
- **Styles**: Tailwind configuration in tailwind.config.css; no separate CSS files unless for global styles
- **Assets**: Store images, icons, and media in src/assets/
- **Hooks**: Custom React hooks go in src/hooks/
- **Utilities**: Helper functions and constants in src/utils/
- **Types**: TypeScript interfaces and types in src/types/

## Performance Constraints
- Implement code splitting for routes using React.lazy and Suspense
- Optimize images and assets (use appropriate formats, compress, lazy load)
- Memoize expensive computations with useMemo and useCallback
- Avoid unnecessary re-renders by using React.memo where beneficial
- Keep bundle size under 2MB gzipped for initial load
- Implement loading states and skeletons for async operations

## Code Quality Requirements
- Write clean, readable, and maintainable code with meaningful variable and function names
- Use TypeScript strictly; avoid 'any' type unless absolutely necessary with proper justification
- Follow functional component patterns; avoid class components
- Implement proper error boundaries for error handling
- Write self-documenting code; add comments only for complex logic or non-obvious solutions
- Keep components small and focused (single responsibility principle)
- Use ESLint and Prettier; resolve all linting errors before committing
- Follow consistent naming conventions (PascalCase for components, camelCase for variables/functions)

## Testing Expectations
- Write unit tests for all custom hooks and utility functions
- Test component rendering and basic interactions
- Aim for minimum 80% code coverage for critical paths
- Use Jest and React Testing Library as the testing framework
- Include both positive and negative test cases
- Mock external dependencies (API calls, etc.) in tests
- Write tests alongside feature development (test-driven approach encouraged)

## Git and Workflow Practices
- Write descriptive commit messages following conventional commits format
- Create feature branches for all new work
- Pull request reviews required for all changes
- Keep dependencies updated regularly
- Address security vulnerabilities promptly