import React from "react";

interface ListElementProps {
  children: React.ReactNode;
  onClick?: React.MouseEventHandler<HTMLLIElement>;
  className?: string;
}

const ListElement: React.FC<ListElementProps> = ({ children, onClick, className = "" }) => {
  return (
    <li className={className} onClick={onClick}>
      {children}
    </li>
  );
};

export default ListElement;
